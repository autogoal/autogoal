from autogoal.grammar import Sampler, GraphGrammar, Graph
from keras.layers import concatenate, Input
from keras.models import Model
from keras.utils import to_categorical


class KerasNeuralNetwork:
    def __init__(
        self,
        grammar: GraphGrammar,
        input_shape=None,
        epochs=10,
        **compile_kwargs
    ):
        self.grammar = grammar
        self._input_shape = input_shape
        self._epochs = epochs
        self._compile_kwargs = compile_kwargs
        self._model = None

    def __repr__(self):
        return (
            "KerasNeuralNetwork(grammar=<...>, input_shape=%r, epochs=%r, compile_kwargs=%r)"
            % (self._input_shape, self._epochs, self._compile_kwargs)
        )

    @property
    def model(self):
        if self._model is None:
            raise TypeError("You need to call `sample` first to generate the model.")

        return self._model

    def sample(self, sampler: Sampler = None):
        if sampler is None:
            sampler = Sampler()

        graph = self.grammar.sample(sampler=sampler)
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
    def fit(self, X, y):
        self._classes = {k: v for k, v in zip(set(y), range(len(y)))}
        self._inverse_classes = {v:k for k,v in self._classes.items()}
        y = [self._classes[yi] for yi in y]
        y = to_categorical(y)
        return super(KerasClassifier, self).fit(X, y)

    def predict(self, X):
        predictions = super(KerasClassifier, self).predict(X)
        predictions = predictions.argmax(axis=-1)
        return [self._inverse_classes[yi] for yi in predictions]
