# coding: utf8

import random

from autogoal.grammar import Block, GraphGrammar, Path
from keras.layers import Concatenate, Dense, Embedding, Input, Softmax
from keras.models import Model


def initializer(cls):
    if cls == Embedding:
        return Embedding(input_dim=1000, output_dim=100)

    if cls == Dense:
        return Dense(units=32)

    return cls()


def main():
    random.seed(0)

    grammar = GraphGrammar()

    grammar.add(Dense, Path(Dense, Dense), initializer=initializer)
    grammar.add(Dense, Block(Dense, Dense), initializer=initializer)
    grammar.add(Softmax, Path(Dense, Softmax), initializer=initializer)

    initial_graph = Path(Embedding, Softmax).make(initializer=initializer)

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
