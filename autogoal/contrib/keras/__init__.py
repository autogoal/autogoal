# coding: utf8

from autogoal.grammar import Sampler, GraphGrammar, Graph
from keras.layers import concatenate, Input
from keras.models import Model


class KerasNeuralNetwork:
    def __init__(self, grammar: GraphGrammar, input_shape=None):
        self.grammar = grammar
        self._input_shape = input_shape

    def sample(self, sampler: Sampler = None):
        if sampler is None:
            sampler = Sampler()

        graph = self.grammar.sample(sampler=sampler)
        model = self._build_nn(graph)
        return model

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

        model = Model(inputs=input_x, outputs=final_ouput)
        return model

    def _build_input(self):
        if self._input_shape is None:
            raise ValueError(
                "You must either provide `input_shape` or redefine `_build_input()`."
            )

        return Input(self._input_shape)

    def _build_output(self, outputs):
        return outputs
