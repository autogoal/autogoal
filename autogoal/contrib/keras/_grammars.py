from autogoal.contrib.keras._generated import *
from autogoal.grammar import GraphGrammar, Path, Block, CfgInitializer, Epsilon


def build_grammar(
    preprocessing=False,
    preprocessing_recurrent=True,
    preprocessing_conv=False,
    reduction=False,
    reduction_recurrent=True,
    reduction_conv=False,
    features=False,
    features_time_distributed=False,
):
    modules = []

    if preprocessing:
        modules.append("PreprocessingModule")

    if reduction:
        modules.append("ReductionModule")

    if features:
        modules.append("FeaturesModule")

    if features_time_distributed:
        modules.append("FeaturesTimeDistributedModule")

    if preprocessing_recurrent and preprocessing_conv:
        raise ValueError("Cannot combine recurrent with convolutional preprocessing.")

    if reduction_recurrent and reduction_conv:
        raise ValueError("Cannot combine recurrent with convolutional reduction modules.")

    if (reduction or features) and features_time_distributed:
        raise ValueError("Cannot combine time-distributed modules with flat modules.")

    if not modules:
        raise ValueError("At least one module must be activated.")

    grammar = GraphGrammar(start=Path(*modules), initializer=CfgInitializer())

    if preprocessing_recurrent:
        grammar.add("PreprocessingModule", "PreprocessingModuleR")
        grammar.add("PreprocessingModuleR", Path("Recurrent", "PreprocessingModuleR"))
        grammar.add("PreprocessingModuleR", Epsilon())
        grammar.add("Recurrent", Seq2SeqLSTM)
        grammar.add("Recurrent", Seq2SeqBiLSTM)

    if preprocessing_conv:
        grammar.add("PreprocessingModule", Path("Convolutional", "PreprocessingModuleC"))
        grammar.add(
            "PreprocessingModuleC", Path("Convolutional", "PreprocessingModuleC")
        )
        grammar.add("PreprocessingModuleC", Epsilon())
        grammar.add("Convolutional", Path(Conv2D, MaxPooling2D))

    if reduction_recurrent:
        grammar.add("ReductionModule", Seq2VecLSTM)
        grammar.add("ReductionModule", Seq2VecBiLSTM)

    if reduction_conv:
        grammar.add("ReductionModule", Flatten)

    if features:
        grammar.add("FeaturesModule", Path("DenseLayer", "FeaturesModule"))
        grammar.add("FeaturesModule", Epsilon())
        grammar.add("DenseLayer", Block(Dense, "DenseLayer"))
        grammar.add("DenseLayer", Path(Dense, "DenseLayer"))
        grammar.add("DenseLayer", Epsilon())

    if features_time_distributed:
        grammar.add(
            "FeaturesTimeDistributedModule",
            Path("TDLayer", "FeaturesTimeDistributedModule"),
        )
        grammar.add("FeaturesTimeDistributedModule", Epsilon())
        grammar.add("TDLayer", Block(TimeDistributed, "TDLayer"))
        grammar.add("TDLayer", Path(TimeDistributed, "TDLayer"))
        grammar.add("TDLayer", Epsilon())

    return grammar
