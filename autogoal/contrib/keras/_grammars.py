from autogoal.contrib.keras._generated import *
from autogoal.grammar import GraphGrammar, Path, Block, CfgInitializer, Epsilon


def sequence_classifier_grammar():
    grammar = GraphGrammar(
        start=Path(
            "PreprocessingModule",
            "ReductionModule",
            "FeaturesModule",
        ),
        initializer=CfgInitializer(),
    )

    grammar.add("PreprocessingModule", Path("Recurrent", "PreprocessingModule"))
    grammar.add("PreprocessingModule", Epsilon())
    grammar.add("Recurrent", Seq2SeqLSTM)
    grammar.add("Recurrent", Seq2SeqBiLSTM)

    grammar.add("ReductionModule", Seq2VecLSTM)
    grammar.add("ReductionModule", Seq2VecBiLSTM)

    grammar.add("FeaturesModule", Path("Layer", "FeaturesModule"))
    grammar.add("FeaturesModule", Epsilon())
    grammar.add("Layer", Block(Dense, "Layer"))
    grammar.add("Layer", Path(Dense, "Layer"))
    grammar.add("Layer", Epsilon())

    return grammar
