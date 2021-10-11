from autogoal.contrib import find_classes
from autogoal.kb import (
    build_pipeline_graph,
    MatrixContinuousDense,
    Supervised,
    VectorCategorical,
    Tensor3,
    Tensor4,
    Seq,
    Label,
)

from autogoal.contrib.keras import (
    KerasClassifier,
    KerasImageClassifier,
    KerasSequenceClassifier,
    KerasSequenceTagger,
)

from autogoal.contrib.keras._generated import (
    Activation,
    Conv1D,
    Conv2D,
    Dense,
    BatchNormalization,
    Dropout,
    Embedding,
    Flatten,
    MaxPooling2D,
    Reshape2D,
    Seq2SeqLSTM,
    Seq2VecBiLSTM,
    Seq2VecLSTM,
    Seq2SeqBiLSTM,
    TimeDistributed,
)

from tensorflow.keras.activations import softmax


def test_matrix_classification_pipeline_uses_keras_classifier():
    pipelines = build_pipeline_graph(
        input_types=(MatrixContinuousDense, Supervised[VectorCategorical]),
        output_type=VectorCategorical,
        registry=find_classes("Keras"),
    )

    nodes = pipelines.nodes()

    assert KerasClassifier in nodes
    assert KerasImageClassifier not in nodes
    assert KerasSequenceClassifier not in nodes
    assert KerasSequenceTagger not in nodes


def test_algorithms_report_correct_types():
    assert KerasClassifier.input_types() == (
        MatrixContinuousDense,
        Supervised[VectorCategorical],
    )
    assert KerasImageClassifier.input_types() == (
        Tensor4,
        Supervised[VectorCategorical],
    )
    assert KerasSequenceClassifier.input_types() == (
        Tensor3,
        Supervised[VectorCategorical],
    )
    assert KerasSequenceTagger.input_types() == (
        Seq[MatrixContinuousDense],
        Supervised[Seq[Seq[Label]]],
    )


def test_keras_layers_from_config():
    activation = Activation(softmax)
    conv1D = Conv1D(32, 3)
    conv2D = Conv2D(32, 3, 0.0005, 0.0005)
    dense = Dense(32)
    batchNormalization = BatchNormalization()
    dropout = Dropout(0.3)
    embedding = Embedding(32)
    flatten = Flatten()
    maxPooling2D = MaxPooling2D()
    rashape = Reshape2D()
    seq2SeqLSTM = Seq2SeqLSTM(32, "linear", "linear", 0.25, 0.25)
    seq2VecBiLSTM = Seq2VecBiLSTM("sum", 32, "linear", "linear", 0.25, 0.25)
    seq2VecLSTM = Seq2VecLSTM(32, "linear", "linear", 0.25, 0.25)
    seq2SeqBiLSTM = Seq2SeqBiLSTM("sum", 32, "linear", "linear", 0.25, 0.25)
    timeDistributed = TimeDistributed(dense)

    layers = [
        activation,
        conv1D,
        conv2D,
        dense,
        batchNormalization,
        dropout,
        embedding,
        flatten,
        maxPooling2D,
        rashape,
        seq2SeqLSTM,
        seq2VecBiLSTM,
        seq2VecLSTM,
        seq2SeqBiLSTM,
        timeDistributed,
    ]

    for layer in layers:
        layer.__class__.from_config(layer.get_config())
