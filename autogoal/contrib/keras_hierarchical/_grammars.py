from autogoal.contrib.keras_hierarchical._generated import *
from autogoal.grammar import GraphGrammar, Path, Block, CfgInitializer, Epsilon


def build_grammar(
    preprocessing_conv2D=False,
    preprocessing_seq=False,
    main_conv2D=False,
    main_seq=False,
    features=False,
    features_td=False,
):
    modules = []

    if preprocessing_conv2D or preprocessing_seq:
        modules.append("PreprocessingModule")

    if main_conv2D or main_seq or features:
        modules.append("MainModule")

    if not main_seq and features_td:
        raise ValueError(
            "Feature Time Distributed module must be combined with Sequential main module"
        )

    if preprocessing_conv2D and preprocessing_seq:
        raise ValueError(
            "Cannot combine Conv2D preprocessing module with Sequential preprocessing module."
        )

    if main_conv2D and main_seq:
        raise ValueError("Cannot combine Conv2D main module with Sequential main module.")

    if not modules:
        raise ValueError("At least one module must be activated.")

    #_______________________________ GRAMMAR ________________________________

    grammar = GraphGrammar(start=Path(*modules), initializer=CfgInitializer())

    # PREPROCESSING
    if preprocessing_conv2D:
        grammar.add("PreprocessingModule", "Conv2D")
        use_conv2d(grammar)
    if preprocessing_seq:
        grammar.add("PreprocessingModule", "LSTMStack")
        use_lstm(grammar)
        grammar.add("PreprocessingModule", "Conv1D")
        use_conv1d(grammar)

    # MAIN
    if main_conv2D:
        grammar.add("MainModule", Path(Flatten, "FeaturesModule"))
        grammar.add("MainModule", GlobalAveragePooling2D)
    if main_seq:
        if features_td:
            grammar.add("MainModule", "FeaturesTDModule")
            use_features_td(grammar)
        grammar.add("MainModule", Path("ReductionRec", "FeaturesModule"))
        grammar.add("MainModule", Path("ReductionConv1D", "FeaturesModule"))
        grammar.add("ReductionRec", Seq2VecLSTM)
        grammar.add("ReductionRec", Seq2VecBiLSTM)
        grammar.add("ReductionConv1D", GlobalAveragePooling1D)
        grammar.add("ReductionConv1D", GlobalMaxPooling1D)
        grammar.add("ReductionConv1D", Flatten)
    if not main_conv2D and not main_seq:
        grammar.add("MainModule", "FeaturesModule")

    # FEATURES
    if features:
        # grammar.add("FeaturesModule", Path(Dropout, "FeaturesModule"))
        grammar.add("FeaturesModule", "DenseBlock")
        use_dense(grammar)

    return grammar


def use_conv2d(grammar):
    grammar.add("Conv2D", Path("Conv2DBlock", "Conv2D"))
    grammar.add("Conv2D", Epsilon())
    grammar.add("Conv2DBlock", Path("Conv2DCells", MaxPooling2D))
    grammar.add("Conv2DBlock", Path("Conv2DCells", MaxPooling2D, Dropout))
    grammar.add("Conv2DCells", Path("Conv2DCell", "Conv2DCells"))
    grammar.add("Conv2DCells", "Conv2DCell")
    grammar.add("Conv2DCell", Path(SeparableConv2D, Activation))
    grammar.add("Conv2DCell", Path(SeparableConv2D, Activation, BatchNormalization))
    grammar.add("Conv2DCell", Path(Conv2D, Activation))
    grammar.add("Conv2DCell", Path(Conv2D, Activation, BatchNormalization))


def use_conv1d(grammar):
    grammar.add("Conv1D", Path("Conv1DCells", "Conv1D"))
    grammar.add("Conv1D", Epsilon())
    grammar.add("Conv1DCells", Path("Conv1DCell", "Conv1DCells"))
    grammar.add("Conv1DCells", "Conv1DCell")
    grammar.add("Conv1DCell", Path(Conv1D, Activation))
    grammar.add("Conv1DCell", Path(Conv1D, Activation, Dropout))


def use_lstm(grammar):
    grammar.add("LSTMStack", Path("Recurrent", "LSTMStack"))
    grammar.add("LSTMStack", Epsilon())
    grammar.add("Recurrent", Seq2SeqLSTM)
    grammar.add("Recurrent", Seq2SeqBiLSTM)


def use_dense(grammar):
    grammar.add("DenseBlock", Path("DenseLayer", "DenseBlock"))
    grammar.add("DenseBlock", Block("DenseLayer", "DenseBlock"))
    grammar.add("DenseBlock", Epsilon())
    grammar.add("DenseLayer", Path(Dense, Activation))
    grammar.add("DenseLayer", Path(Dense, Activation, Dropout))


def use_features_td(grammar):
    grammar.add(
        "FeaturesTDModule", Path("TDLayer", "FeaturesTDModule"),
    )
    grammar.add("FeaturesTDModule", Epsilon())
    grammar.add("TDLayer", Block(TimeDistributed(Dense), "TDLayer"))
    grammar.add("TDLayer", Path(TimeDistributed(Dense), "TDLayer"))
    grammar.add("TDLayer", Epsilon())
    return grammar
