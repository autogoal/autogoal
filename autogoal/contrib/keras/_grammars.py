from autogoal.contrib.keras._generated import *
from autogoal.grammar import GraphGrammar, Path, Block, CfgInitializer


def sequence_classifier_grammar():
    grammar = GraphGrammar(
        start=Path(
            "EncodingModule",
            "FlattenModule",
            "FeaturesModule",
        ),
        initializer=CfgInitializer(),
    )

    grammar.add("EncodingModule", Seq2SeqLSTM)

    grammar.add("FlattenModule", Seq2VecLSTM)

    grammar.add("FeaturesModule", Dense)

    # # productions for Preprocessing
    # grammar.add("PreprocessingModule", Path(Dense, Reshape2D))
    
    # # productions for Reduction
    # grammar.add("ReductionModule", "ConvModule")
    # grammar.add("ReductionModule", "DenseModule")
    # grammar.add("ConvModule", Path(Conv1D, MaxPool1D, "ConvModule"))
    # grammar.add("ConvModule", Path(Conv1D, MaxPool1D))

    # # productions for Features
    # grammar.add("FeaturesModule", Path(Flatten, "DenseModule"))
    # # TODO: Attention

    # # productions for Classification
    # grammar.add("ClassificationModule", "DenseModule")

    # # productions to expand Dense layers
    # grammar.add("DenseModule", Path(Dense, "DenseModule"))
    # grammar.add("DenseModule", Block(Dense, "DenseModule"))
    # grammar.add("DenseModule", Dense)

    return grammar
