from autogoal.contrib import find_classes
from autogoal.kb import (
    build_pipeline_graph,
    MatrixContinuousDense,
    Supervised,
    VectorCategorical,
    Tensor3,
    Tensor4,
    Seq,
    Postag,
)

from autogoal.contrib.keras import (
    KerasClassifier,
    KerasImageClassifier,
    KerasSequenceClassifier,
    KerasSequenceTagger,
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
        Supervised[Seq[Seq[Postag]]],
    )
