from typing import Optional

from autogoal.contrib.keras._grammars import sequence_classifier_grammar
from autogoal.grammar import Graph, GraphGrammar, Sampler
from autogoal.kb import CategoricalVector, Tensor3, Tuple
from keras.layers import Dense, Input, concatenate
from keras.models import Model
from keras.utils import to_categorical


class KerasNeuralNetwork:
    def __init__(
        self, grammar: GraphGrammar, input_shape=None, epochs=10, **compile_kwargs
    ):
        self.grammar = grammar
        self._input_shape = input_shape
        self._epochs = epochs
        self._compile_kwargs = compile_kwargs
        self._model: Optional[Model] = None
        self._mode = "train"

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
        parameters = self._model.count_params() if self._model else None

        return (
            "KerasNeuralNetwork(parameters=%i, input_shape=%r, epochs=%r, compile_kwargs=%r)"
            % (parameters, self._input_shape, self._epochs, self._compile_kwargs)
        )

    @property
    def model(self):
        if self._model is None:
            raise TypeError("You need to call `sample` first to generate the model.")

        return self._model

    def sample(self, sampler: Sampler = None, max_iterations=100):
        if sampler is None:
            sampler = Sampler()

        graph = self.grammar.sample(sampler=sampler, max_iterations=max_iterations)
        self._build_nn(graph)

    def _build_nn(self, graph: Graph):
        input_x = self._build_input()

        def build_model(layer, _, previous_layers):
            if not previous_layers:
                return layer(input_x)

            if len(previous_layers) > 1:
                incoming = concatenate(previous_layers)
            else:
                incoming = previous_layers[0]

            return layer(incoming)

        output_y = graph.apply(build_model)
        final_ouput = self._build_output(output_y)

        if "optimizer" not in self._compile_kwargs:
            self._compile_kwargs["optimizer"] = "adam"

        self._model = Model(inputs=input_x, outputs=final_ouput)
        self._model.compile(**self._compile_kwargs)

    def _build_input(self):
        if self._input_shape is None:
            raise ValueError(
                "You must either provide `input_shape` or redefine `_build_input()`."
            )

        return Input(self._input_shape)

    def _build_output(self, outputs):
        return outputs

    def fit(self, X, y):
        self.model.fit(x=X, y=y, epochs=self._epochs)

    def predict(self, X):
        return self.model.predict(X)


class KerasClassifier(KerasNeuralNetwork):
    def __init__(self, classes=2, *args, **kwargs):
        self._num_classes = classes
        self._classes = None
        super().__init__(*args, **kwargs)

    def _build_output(self, outputs):
        if len(outputs) > 1:
            outputs = concatenate(outputs)
        else:
            outputs = outputs[0]

        if "loss" not in self._compile_kwargs:
            self._compile_kwargs["loss"] = "categorical_crossentropy"

        return Dense(units=self._num_classes, activation="softmax")(outputs)

    def fit(self, X, y):
        self._classes = {k: v for k, v in zip(set(y), range(len(y)))}

        if len(self._classes) != self._num_classes:
            raise ValueError(
                "Expected %i different classes but got: %r"
                % (self._num_classes, list(self._classes))
            )

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


class KerasSequenceClassifier(KerasClassifier):
    def __init__(self, embedding_size=768, *args, **kwargs):
        super().__init__(
            grammar=sequence_classifier_grammar(),
            input_shape=(None, embedding_size),
            *args,
            **kwargs
        )

    def run(self, input: Tuple(Tensor3(), CategoricalVector())) -> CategoricalVector():
        return super().run(input)
