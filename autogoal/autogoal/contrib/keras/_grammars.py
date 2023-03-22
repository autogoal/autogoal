from autogoal.contrib.keras._generated import *
from autogoal.grammar import GraphGrammar, Path, Block, CfgInitializer, Epsilon


class Module:
    def make_top_level(self, top_level: list):
        pass

    def add_productions(self, grammar: GraphGrammar):
        raise NotImplementedError()


class Modules:
    class Preprocessing:
        class Recurrent(Module):
            def make_top_level(self, top_level):
                if "PreprocessingModule" not in top_level:
                    top_level.append("PreprocessingModule")

            def add_productions(self, grammar: GraphGrammar):
                grammar.add("PreprocessingModule", "RecurrentModule")

                grammar.add("RecurrentModule", Path("RecurrentCell", "RecurrentModule"))
                grammar.add("RecurrentModule", "RecurrentCell")

                grammar.add("RecurrentCell", Seq2SeqLSTM)
                grammar.add("RecurrentCell", Seq2SeqBiLSTM)

        class Conv2D(Module):
            def make_top_level(self, top_level):
                if "PreprocessingModule" not in top_level:
                    top_level.append("PreprocessingModule")

            def add_productions(self, grammar: GraphGrammar):
                grammar.add("PreprocessingModule", Path("Conv2DModule", Flatten))

                grammar.add("Conv2DModule", Path("Conv2DBlock", "Conv2DModule"))
                grammar.add("Conv2DModule", "Conv2DBlock")

                grammar.add("Conv2DBlock", Path("Conv2DCells", MaxPooling2D))
                grammar.add("Conv2DBlock", Path("Conv2DCells", MaxPooling2D, Dropout))

                grammar.add("Conv2DCells", Path("Conv2DCell", "Conv2DCells"))
                grammar.add("Conv2DCells", Path("Conv2DCell"))

                grammar.add("Conv2DCell", Path(Conv2D, Activation, BatchNormalization))
                grammar.add("Conv2DCell", Path(Conv2D, Activation))

    class Features:
        class Dense(Module):
            def make_top_level(self, top_level):
                if "FeaturesModule" not in top_level:
                    top_level.append("FeaturesModule")

            def add_productions(self, grammar):
                grammar.add("FeaturesModule", Path("DenseModule", "FeaturesModule"))
                grammar.add("FeaturesModule", Epsilon())

                grammar.add("DenseModule", Block("DenseCell", "DenseModule"))
                grammar.add("DenseModule", Path("DenseCell", "DenseModule"))
                grammar.add("DenseModule", Epsilon())
                grammar.add("DenseCell", Path(Dense, Activation, Dropout))
                grammar.add("DenseCell", Path(Dense, Activation))


def generate_grammar(*modules):
    top_level = []

    for mod in modules:
        mod.make_top_level(top_level)

    grammar = GraphGrammar(start=Path(*top_level), initializer=CfgInitializer())

    for mod in modules:
        mod.add_productions(grammar)

    return grammar


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
        raise ValueError(
            "Cannot combine recurrent with convolutional reduction modules."
        )

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
        grammar.add(
            "PreprocessingModule", Path("Convolutional", "PreprocessingModuleC")
        )
        grammar.add(
            "PreprocessingModuleC", Path("Convolutional", "PreprocessingModuleC")
        )
        grammar.add("PreprocessingModuleC", Epsilon())
        grammar.add("Convolutional", Path(Conv2D, MaxPooling2D))
        grammar.add("Convolutional", Path(Conv2D, MaxPooling2D, Dropout))

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
        grammar.add("DenseLayer", Path(Dense, Dropout, "DenseLayer"))
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
