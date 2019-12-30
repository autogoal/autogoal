# coding: utf8


from autogoal.grammar import GraphGrammar, Block, Path, Graph
from keras.layers import Input
from keras.layers import (
    Dense,
    Embedding,
    LSTM,
    Conv1D,
    Conv2D,
    MaxPool1D,
    MaxPool2D,
    Dropout,
    concatenate,
    Flatten,
)
from keras.models import Model


class PreprocessingModule:
    pass


class ReductionModule:
    pass


class FeaturesModule:
    pass


class ClassificationModule:
    pass


def custom_init(cls):
    if cls == Embedding:
        return cls(input_dim=1000, output_dim=32)

    if cls == Dense:
        return cls(units=128)

    if cls == Conv1D or cls == Conv2D:
        return cls(filters=5, kernel_size=7)

    return cls()


def build_grammar():
    grammar = GraphGrammar(
        non_terminals=[
            PreprocessingModule,
            ReductionModule,
            FeaturesModule,
            ClassificationModule,
        ]
    )

    # productions for Preprocessing
    grammar.add(PreprocessingModule, Embedding, initializer=custom_init)
    # grammar.add(PreprocessingModule, Dense, initializer=custom_init)
    # grammar.add(PreprocessingModule, Path(Embedding, LSTM))
    # TODO: Bert

    # productions for Reduction
    grammar.add(ReductionModule, Conv1D, initializer=custom_init)
    # grammar.add(ReductionModule, Conv2D, initializer=custom_init)
    grammar.add(Conv1D, Path(Conv1D, Dense), initializer=custom_init)
    grammar.add(Conv2D, Path(Conv2D, Dense), initializer=custom_init)

    # productions for Features
    grammar.add(FeaturesModule, Dense, initializer=custom_init)
    # TODO: Attention

    # productions for Classification
    grammar.add(ClassificationModule, Dense, initializer=custom_init)

    # productions to expand Dense layers
    grammar.add(Dense, Path(Dense, Dense), initializer=custom_init)
    grammar.add(Dense, Block(Dense, Dense), initializer=custom_init)

    return grammar


def build_graph(grammar: GraphGrammar):
    # instantiate the initial graph with all abstract modules
    initial_graph = Path(
        PreprocessingModule, ReductionModule, FeaturesModule, ClassificationModule
    ).make()

    return grammar.expand(initial_graph, max_iters=10)


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
    classes = Dense(units=4, activation="softmax")(Flatten()(output_y))

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
