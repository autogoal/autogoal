# coding: utf8


from keras.layers import (
    LSTM,
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Input,
    MaxPool1D,
    Reshape,
    concatenate,
)
from keras.models import Model
from keras.utils import plot_model

from autogoal.grammar import Block, Graph, GraphGrammar, Path


class Reshape2D(Reshape):
    def __init__(self):
        super(Reshape2D, self).__init__(target_shape=(-1, 1))


def custom_init(cls):
    if cls == Embedding:
        return cls(input_dim=1000, output_dim=32)

    if cls == Dense:
        return cls(units=128)

    if cls == Conv1D:
        return cls(filters=5, kernel_size=7, padding="causal")

    return cls()


def build_grammar():
    grammar = GraphGrammar()

    # productions for Preprocessing
    grammar.add("PreprocessingModule", Embedding, initializer=custom_init)
    grammar.add("PreprocessingModule", Path(Dense, Reshape2D), initializer=custom_init)
    # grammar.add("PreprocessingModule", Path(Embedding, LSTM))
    # TODO: Bert

    # productions for Reduction
    grammar.add("ReductionModule", "ConvModule", initializer=custom_init)
    grammar.add("ReductionModule", "DenseModule", initializer=custom_init)
    grammar.add(
        "ConvModule", Path(Conv1D, MaxPool1D, "ConvModule"), initializer=custom_init
    )
    grammar.add("ConvModule", Path(Conv1D, MaxPool1D), initializer=custom_init)
    # grammar.add("ReductionModule", Path(Conv2D, MaxPool2D, "DenseModule"), initializer=custom_init)

    # productions for Features
    grammar.add("FeaturesModule", Path(Flatten, "DenseModule"), initializer=custom_init)
    # TODO: Attention

    # productions for Classification
    grammar.add("ClassificationModule", "DenseModule", initializer=custom_init)

    # productions to expand Dense layers
    grammar.add("DenseModule", Path(Dense, "DenseModule"), initializer=custom_init)
    grammar.add("DenseModule", Block(Dense, "DenseModule"), initializer=custom_init)
    grammar.add("DenseModule", Dense, initializer=custom_init)

    return grammar


def build_graph(grammar: GraphGrammar):
    # instantiate the initial graph with all abstract modules
    initial_graph = Path(
        "PreprocessingModule",
        "ReductionModule",
        "FeaturesModule",
        "ClassificationModule",
    ).make()

    return grammar.expand(initial_graph, max_iters=100)


def build_nn(graph: Graph):
    input_x = Input((1000,))

    def build_model(layer, _, previous_layers):
        if not previous_layers:
            return layer(input_x)

        if len(previous_layers) > 1:
            incoming = concatenate(previous_layers)
        else:
            incoming = previous_layers[0]

        return layer(incoming)

    output_y = graph.apply(build_model)
    classes = Dense(units=4, activation="softmax")(output_y)

    model = Model(inputs=input_x, outputs=classes)
    model.compile("rmsprop", loss="categorical_crossentropy")

    return model


def main():
    grammar = build_grammar()
    graph = build_graph(grammar)
    model = build_nn(graph)

    model.summary()


if __name__ == "__main__":
    main()
