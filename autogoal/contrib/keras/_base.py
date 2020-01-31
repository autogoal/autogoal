from typing import Optional

from autogoal.contrib.keras._grammars import build_grammar
from autogoal.grammar import Graph, GraphGrammar, Sampler
from autogoal.kb import CategoricalVector, Tensor3, Tuple, MatrixContinuousDense, List
from keras.layers import Dense, Input, concatenate, TimeDistributed
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping


class KerasNeuralNetwork:
    def __init__(
        self, grammar: GraphGrammar, epochs=10, validation_split=0.1, **compile_kwargs
    ):
        self.grammar = grammar
        self._epochs = epochs
        self._compile_kwargs = compile_kwargs
        self._model: Optional[Model] = None
        self._mode = "train"
        self._graph = None
        self._validation_split = validation_split

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

    @property
    def model(self):
        if self._model is None:
            raise TypeError("You need to call `fit` first to generate the model.")

        return self._model

    def sample(self, sampler: Sampler = None, max_iterations=100):
        if sampler is None:
            sampler = Sampler()

        self._graph = self.grammar.sample(
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
            self._compile_kwargs["optimizer"] = "adam"

        self._model = Model(inputs=input_x, outputs=final_ouput)
        self._model.compile(**self._compile_kwargs)

    def _build_input(self, X):
        raise NotImplementedError()

    def _build_output(self, outputs, y):
        return outputs

    def fit(self, X, y):
        self._build_nn(self._graph, X, y)
        self.model.fit(
            x=X,
            y=y,
            epochs=self._epochs,
            callbacks=[EarlyStopping()],
            validation_split=self._validation_split,
        )

    def predict(self, X):
        return self.model.predict(X)


class KerasClassifier(KerasNeuralNetwork):
    def __init__(self, **kwargs):
        self._classes = None
        self._num_classes = None
        super().__init__(grammar=self._build_grammar(), **kwargs)

    def _build_grammar(self):
        return build_grammar(features=True)

    def _build_input(self, X):
        return Input(shape=(X.shape[1],))

    def _build_output_layer(self, y):
        self._num_classes = y.shape[1]

        if "loss" not in self._compile_kwargs:
            self._compile_kwargs["loss"] = "categorical_crossentropy"

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


class KerasSequenceClassifier(KerasClassifier):
    def _build_grammar(self):
        return build_grammar(preprocessing=True, reduction=True, features=True)

    def _build_input(self, X):
        return Input(shape=(None, X.shape[2]))

    def run(self, input: Tuple(Tensor3(), CategoricalVector())) -> CategoricalVector():
        return super().run(input)


class KerasSequenceTagger(KerasClassifier):
    def _build_grammar(self):
        return build_grammar(preprocessing=True, features_time_distributed=True)

    def _build_input(self, X):
        return Input(shape=(None, X.shape[2]))

    def _build_output(self, outputs, y):
        if len(outputs) > 1:
            outputs = concatenate(outputs)
        else:
            outputs = outputs[0]

        return TimeDistributed(super()._build_output_layer(y))(outputs)

    def run(self, input: Tuple(Tensor3(), List(CategoricalVector()))) -> List(CategoricalVector()):
        return super().run(input)
