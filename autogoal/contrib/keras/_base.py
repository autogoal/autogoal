from typing import Optional

import collections
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from tensorflow.keras.layers import Dense, Input, TimeDistributed, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator as _ImageDataGenerator,
)

from autogoal.contrib.keras._grammars import build_grammar, generate_grammar, Modules
from autogoal.grammar import (
    Graph,
    GraphGrammar,
    Sampler,
    Boolean,
    Categorical,
    Continuous,
    Discrete,
)
from autogoal.kb import (
    CategoricalVector,
    List,
    MatrixContinuousDense,
    Postag,
    Tensor3,
    Tensor4,
    Tuple,
)
from autogoal.utils import nice_repr


@nice_repr
class KerasNeuralNetwork:
    def __init__(
        self,
        grammar: GraphGrammar,
        optimizer: Categorical("sgd", "adam", "rmsprop"),
        epochs=10,
        early_stop=3,
        validation_split=0.1,
        **compile_kwargs,
    ):
        self.optimizer = optimizer
        self._grammar = grammar
        self._epochs = epochs
        self._compile_kwargs = compile_kwargs
        self._model: Optional[Model] = None
        self._mode = "train"
        self._graph = None
        self._validation_split = validation_split
        self._early_stop = early_stop

    def train(self):
        self._mode = "train"

    def eval(self):
        self._mode = "eval"

    def run(self, input):
        X, y = input
        if self._mode == "train":
            self.fit(X, y)
            return y
        if self._mode == "eval":
            return self.predict(X)

        assert False, "Invalid mode %s" % self._mode

    def __repr__(self):
        nodes = len(self._graph.nodes) if self._graph is not None else None
        return f"{self.__class__.__name__}(nodes={nodes}, compile_kwargs={self._compile_kwargs})"

    def __nice_repr_hook__(self, names, values):
        graph = []

        if self._graph is not None:
            for node, _ in self._graph.build_order():
                graph.append(node)

        names.append("graph")
        values.append(graph)

    @property
    def model(self):
        if self._model is None:
            raise TypeError("You need to call `fit` first to generate the model.")

        return self._model

    def sample(self, sampler: Sampler = None, max_iterations=100):
        if sampler is None:
            sampler = Sampler()

        self._graph = self._grammar.sample(
            sampler=sampler, max_iterations=max_iterations
        )
        return self

    def _build_nn(self, graph: Graph, X, y):
        input_x = self._build_input(X)

        def build_model(layer, _, previous_layers):
            if not previous_layers:
                return layer(input_x)

            if len(previous_layers) > 1:
                incoming = concatenate(previous_layers)
            else:
                incoming = previous_layers[0]

            return layer(incoming)

        output_y = graph.apply(build_model) or [input_x]
        final_ouput = self._build_output(output_y, y)

        if "optimizer" not in self._compile_kwargs:
            self._compile_kwargs["optimizer"] = self.optimizer

        self._model = Model(inputs=input_x, outputs=final_ouput)
        self._model.compile(**self._compile_kwargs)

    def _build_input(self, X):
        raise NotImplementedError()

    def _build_output(self, outputs, y):
        return outputs

    def fit(self, X, y, **kwargs):
        if self._graph is None:
            raise TypeError("You must call `sample` to generate the internal model.")

        self._build_nn(self._graph, X, y)
        self.model.summary()
        self._fit_model(X, y, **kwargs)

    def _fit_model(self, X, y, **kwargs):
        self.model.fit(
            x=X,
            y=y,
            epochs=self._epochs,
            callbacks=[
                EarlyStopping(patience=self._early_stop, restore_best_weights=True),
                TerminateOnNaN(),
            ],
            validation_split=self._validation_split,
            **kwargs,
        )

    def predict(self, X):
        return self.model.predict(X)


class KerasClassifier(KerasNeuralNetwork):
    def __init__(
        self, optimizer: Categorical("sgd", "adam", "rmsprop"), grammar=None, **kwargs
    ):
        self._classes = None
        self._num_classes = None
        super().__init__(
            grammar=grammar or self._build_grammar(), optimizer=optimizer, **kwargs
        )

    def _build_grammar(self):
        return build_grammar(features=True)

    def _build_input(self, X):
        return Input(shape=(X.shape[1],))

    def _build_output_layer(self, y):
        self._num_classes = y.shape[1]

        if "loss" not in self._compile_kwargs:
            self._compile_kwargs["loss"] = "categorical_crossentropy"
            self._compile_kwargs["metrics"] = ["accuracy"]

        return Dense(units=self._num_classes, activation="softmax")

    def _build_output(self, outputs, y):
        if len(outputs) > 1:
            outputs = concatenate(outputs)
        else:
            outputs = outputs[0]

        return self._build_output_layer(y)(outputs)

    def fit(self, X, y):
        self._classes = {k: v for k, v in zip(set(y), range(len(y)))}
        self._inverse_classes = {v: k for k, v in self._classes.items()}
        y = [self._classes[yi] for yi in y]
        y = to_categorical(y)
        return super(KerasClassifier, self).fit(X, y)

    def predict(self, X):
        if self._classes is None:
            raise TypeError(
                "You must call `fit` before `predict` to learn class mappings."
            )

        predictions = super(KerasClassifier, self).predict(X)
        predictions = predictions.argmax(axis=-1)
        return [self._inverse_classes[yi] for yi in predictions]

    def run(
        self, input: Tuple(MatrixContinuousDense(), CategoricalVector())
    ) -> CategoricalVector():
        return super().run(input)


@nice_repr
class KerasImagePreprocessor(_ImageDataGenerator):
    """Augment a dataset of images by making changes to the original training set.

    Applies standard dataset augmentation strategies, such as rotating,
    scaling and fliping the image.
    Uses the `ImageDataGenerator` class from keras.

    The parameter `grow_size` determines how many new images will be created for each original image.
    The remaining parameters are passed to `ImageDataGenerator`.
    """

    def __init__(
        self,
        featurewise_center: Boolean(),
        samplewise_center: Boolean(),
        featurewise_std_normalization: Boolean(),
        samplewise_std_normalization: Boolean(),
        rotation_range: Discrete(0, 15),
        width_shift_range: Continuous(0, 0.25),
        height_shift_range: Continuous(0, 0.25),
        shear_range: Continuous(0, 15),
        zoom_range: Continuous(0, 0.25),
        horizontal_flip: Boolean(),
        vertical_flip: Boolean(),
    ):
        super().__init__(
            featurewise_center=featurewise_center,
            samplewise_center=samplewise_center,
            featurewise_std_normalization=featurewise_std_normalization,
            samplewise_std_normalization=samplewise_std_normalization,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
        )


class KerasImageClassifier(KerasClassifier):
    def __init__(
        self,
        preprocessor: KerasImagePreprocessor,
        optimizer: Categorical("sgd", "adam", "rmsprop"),
        **kwargs,
    ):
        self.preprocessor = preprocessor
        super().__init__(optimizer=optimizer, **kwargs)

    def _build_grammar(self):
        return generate_grammar(
            Modules.Preprocessing.Conv2D(),
            # Modules.Features.Dense()
        )

    def _fit_model(self, X, y, **kwargs):
        self.preprocessor.fit(X)
        batch_size = 64
        validation_size = min(int(0.25 * len(X)), 10 * batch_size)

        Xtrain, Xvalid = X[:-validation_size], X[-validation_size:]
        ytrain, yvalid = y[:-validation_size], y[-validation_size:]

        self.model.fit(
            self.preprocessor.flow(Xtrain, ytrain, batch_size=batch_size),
            steps_per_epoch=len(Xtrain) // batch_size,
            epochs=self._epochs,
            callbacks=[
                EarlyStopping(patience=self._early_stop, restore_best_weights=True),
                TerminateOnNaN(),
            ],
            validation_data=(Xvalid, yvalid),
            **kwargs,
        )

    def run(self, input: Tuple(Tensor4(), CategoricalVector())) -> CategoricalVector():
        return super().run(input)

    def _build_input(self, X):
        return Input(shape=X.shape[1:])


class KerasSequenceClassifier(KerasClassifier):
    def _build_grammar(self):
        return build_grammar(preprocessing=True, reduction=True, features=True)

    def _build_input(self, X):
        return Input(shape=(None, X.shape[2]))

    def run(self, input: Tuple(Tensor3(), CategoricalVector())) -> CategoricalVector():
        return super().run(input)


# !!! warning
# These imports are taken from <https://github.com/tensorflow/addons/pull/377>
# since the CRF layer has not yet landed in tensorflow
# TODO: Make sure to replace this when tensorflow merges this commit
from autogoal.contrib.keras._crf import CRF
from autogoal.contrib.keras._crf import crf_loss


class KerasSequenceTagger(KerasNeuralNetwork):
    def __init__(
        self,
        decode: Categorical("dense", "crf"),
        optimizer: Categorical("sgd", "adam", "rmsprop"),
        grammar=None,
        **kwargs,
    ):
        self._classes = None
        self._num_classes = None

        if decode not in ["dense", "crf"]:
            raise ValueError(f"Invalid decode={decode}")

        self.decode = decode
        super().__init__(
            grammar=grammar or self._build_grammar(), optimizer=optimizer, **kwargs
        )

    def _build_grammar(self):
        from autogoal.grammar._graph import Epsilon

        return GraphGrammar(Epsilon())

    def _build_input(self, X):
        return Input(shape=(None, X[0].shape[-1]))

    def _build_output(self, outputs, y):
        if "loss" not in self._compile_kwargs:
            self._compile_kwargs["metrics"] = ["accuracy"]

            if self.decode == "dense":
                self._compile_kwargs["loss"] = "categorical_crossentropy"
            elif self.decode == "crf":
                self._compile_kwargs["loss"] = crf_loss

        if len(outputs) > 1:
            outputs = concatenate(outputs)
        else:
            outputs = outputs[0]

        if self.decode == "dense":
            dense = Dense(units=len(self._classes), activation="softmax")
            return TimeDistributed(dense)(outputs)
        elif self.decode == "crf":
            crf = CRF(units=len(self._classes))
            return crf(outputs)

    def fit(self, X, y):
        distinct_classes = set(x for yi in y for x in yi)

        self._classes = {
            k: v for k, v in zip(distinct_classes, range(len(distinct_classes)))
        }
        self._inverse_classes = {v: k for k, v in self._classes.items()}

        y = [[self._classes[x] for x in yi] for yi in y]
        return super().fit(X, y)

    def _encode(self, xi, yi):
        if self.decode == "dense":
            yi = to_categorical(yi, len(self._classes))
            
        return (
            np.expand_dims(xi, axis=0),
            np.expand_dims(yi, axis=0),
        )
        
    def _generate_batches(self, X, y):
        while True:
            for xi, yi in zip(X, y):
                yield self._encode(xi, yi)

    def _fit_model(self, X, y, **kwargs):
        self.model.fit(
            self._generate_batches(X, y),
            steps_per_epoch=len(X),
            epochs=self._epochs,
            callbacks=[
                EarlyStopping(
                    patience=self._early_stop,
                    restore_best_weights=True,
                    monitor="accuracy",
                ),
                TerminateOnNaN(),
            ],
            **kwargs,
        )

    def _decode(self, predictions):
        if self.decode == "dense":
            predictions = [pr.argmax(axis=-1) for pr in predictions]
            return [[self._inverse_classes[x] for x in yi[0]] for yi in predictions]
        elif self.decode == "crf":
            return [[self._inverse_classes[x] for x in yi[0]] for yi in predictions]

    def predict(self, X):
        if self._classes is None:
            raise TypeError(
                "You must call `fit` before `predict` to learn class mappings."
            )

        predictions = [self.model.predict(np.expand_dims(xi, axis=0)) for xi in X]
        return self._decode(predictions)

    def run(
        self, input: Tuple(List(MatrixContinuousDense()), List(List(Postag())))
    ) -> List(List(Postag())):
        return super().run(input)
