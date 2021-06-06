from autogoal.experimental.audio_command_recognition.kb._semantics import AudioFeatures
from autogoal.experimental.audio_command_recognition.keras._base import (
    KerasAudioClassifier,
)
from autogoal.kb import Seq, Supervised, VectorCategorical, build_pipeline_graph
from autogoal.contrib import find_classes


def test_matrix_classification_pipeline_uses_keras_classifier():
    pipelines = build_pipeline_graph(
        input_types=(Seq[AudioFeatures], Supervised[VectorCategorical]),
        output_type=VectorCategorical,
        registry=find_classes("Keras") + [KerasAudioClassifier],
    )

    nodes = pipelines.nodes()

    assert KerasAudioClassifier in nodes


def test_algorithm_correct_types():
    assert KerasAudioClassifier.input_types() == (
        Seq[AudioFeatures],
        Supervised[VectorCategorical],
    )
