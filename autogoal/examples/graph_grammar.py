# coding: utf8

import networkx as nx
import random

from keras.layers import Input, Dense, Embedding, Softmax, Concatenate
from keras.models import Model

from autogoal.grammar import GraphGrammar, Path, Block
# from autogoal.ontology._generated._keras import (
#     DenseLayer,
#     ConcatenateLayer,
#     SoftmaxLayer,
#     EmbeddingLayer,
# )


def init_factory(cls):
    if cls == Embedding:
        return dict(input_dim=1000, output_dim=100)

    if cls == Dense:
        return dict(units=32)

    return dict()


def main():
    random.seed(0)

    grammar = GraphGrammar()

    grammar.add(Dense, Path(Dense, Dense), init_factory=init_factory)
    grammar.add(Dense, Block(Dense, Dense), init_factory=init_factory)
    grammar.add(Softmax, Path(Dense, Softmax), init_factory=init_factory)

    initial_graph = Path(Embedding, Softmax).make(
        init_factory=init_factory
    )

    graph = grammar.expand(initial_graph, max_iters=5)

    input_x = Input((1000,))

    def build_model(layer, _, previous_layers):
        if not previous_layers:
            return layer(input_x)

        if len(previous_layers) > 1:
            incoming = Concatenate()(previous_layers)
        else:
            incoming = previous_layers[0]

        return layer(incoming)

    output_y = graph.apply(build_model)

    model = Model(inputs=input_x, outputs=output_y)
    model.summary()


if __name__ == "__main__":
    main()
