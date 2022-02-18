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
    def test_from_config(layer):
        layer.__class__.from_config(layer.get_config())

    activation = Activation("softmax")
    test_from_config(activation)
    conv1D = Conv1D(32, 3)
    test_from_config(conv1D)
    conv2D = Conv2D(32, 3, 0.0005, 0.0005)
    test_from_config(conv2D)
    dense = Dense(32)
    test_from_config(dense)
    batchNormalization = BatchNormalization()
    test_from_config(batchNormalization)
    dropout = Dropout(0.3)
    test_from_config(dropout)
    embedding = Embedding(32)
    test_from_config(embedding)
    flatten = Flatten()
    test_from_config(flatten)
    maxPooling2D = MaxPooling2D()
    test_from_config(maxPooling2D)
    rashape = Reshape2D()
    test_from_config(rashape)
    seq2SeqLSTM = Seq2SeqLSTM(32, "linear", "linear", 0.25, 0.25)
    test_from_config(seq2SeqLSTM)
    seq2VecBiLSTM = Seq2VecBiLSTM("sum", 32, "linear", "linear", 0.25, 0.25)
    test_from_config(seq2VecBiLSTM)
    seq2VecLSTM = Seq2VecLSTM(32, "linear", "linear", 0.25, 0.25)
    test_from_config(seq2VecLSTM)
    seq2SeqBiLSTM = Seq2SeqBiLSTM("sum", 32, "linear", "linear", 0.25, 0.25)
    test_from_config(seq2SeqBiLSTM)
    timeDistributed = TimeDistributed(dense)
    test_from_config(timeDistributed)
