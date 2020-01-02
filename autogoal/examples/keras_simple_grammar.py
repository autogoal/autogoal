# coding: utf8


from keras.layers import (
    LSTM,
    Conv1D as _Conv1D,
    Dense as _Dense,
    Dropout,
    Embedding as _Embedding,
    Flatten,
    Input,
    MaxPool1D,
    Reshape,
    concatenate,
)
from keras.models import Model
from keras.utils import plot_model

from autogoal.grammar import (
    Block,
    Graph,
    GraphGrammar,
    Path,
    CfgInitializer,
    Discrete,
    Categorical,
)


class Reshape2D(Reshape):
    def __init__(self):
        super(Reshape2D, self).__init__(target_shape=(-1, 1))


class Embedding(_Embedding):
    def __init__(self, output_dim: Discrete(32, 128)):
        super(Embedding, self).__init__(input_dim=1000, output_dim=output_dim)


class Dense(_Dense):
    def __init__(self, units: Discrete(128, 1024), **kwargs):
        super(Dense, self).__init__(units=units, **kwargs)


class Conv1D(_Conv1D):
    def __init__(self, filters: Discrete(5, 20), kernel_size: Categorical(3, 5, 7)):
        super(Conv1D, self).__init__(
            filters=filters, kernel_size=kernel_size, padding="causal"
        )


def build_grammar():
    grammar = GraphGrammar(
        start=Path(
            "PreprocessingModule",
            "ReductionModule",
            "FeaturesModule",
            "ClassificationModule",
        ),
        initializer=CfgInitializer(),
    )

    # productions for Preprocessing
    grammar.add("PreprocessingModule", Embedding)
    grammar.add("PreprocessingModule", Path(Dense, Reshape2D))
    # grammar.add("PreprocessingModule", Path(Embedding, LSTM))
    # TODO: Bert

    # productions for Reduction
    grammar.add("ReductionModule", "ConvModule")
    grammar.add("ReductionModule", "DenseModule")
    grammar.add("ConvModule", Path(Conv1D, MaxPool1D, "ConvModule"))
    grammar.add("ConvModule", Path(Conv1D, MaxPool1D))
    # grammar.add("ReductionModule", Path(Conv2D, MaxPool2D, "DenseModule"))

    # productions for Features
    grammar.add("FeaturesModule", Path(Flatten, "DenseModule"))
    # TODO: Attention

    # productions for Classification
    grammar.add("ClassificationModule", "DenseModule")

    # productions to expand Dense layers
    grammar.add("DenseModule", Path(Dense, "DenseModule"))
    grammar.add("DenseModule", Block(Dense, "DenseModule"))
    grammar.add("DenseModule", Dense)

    return grammar


def build_graph(grammar: GraphGrammar):
    # instantiate the initial graph with all abstract modules
    return grammar.sample(max_iterations=100)


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
