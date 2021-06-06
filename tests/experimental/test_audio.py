from autogoal.kb import Seq, Supervised, VectorCategorical, build_pipeline_graph
from autogoal.contrib import find_classes
from autogoal.experimental.audio_command_recognition.audio import (
    AudioClassifier,
    AudioCommandPreprocessor,
)
from autogoal.experimental.audio_command_recognition.audio._generated import (
    AudioCommandReader,
)
from autogoal.experimental.audio_command_recognition.keras._base import (
    KerasAudioClassifier,
)
from autogoal.experimental.audio_command_recognition.kb._semantics import (
    AudioFile,
    AudioFeatures,
    AudioCommand,
)


def test_matrix_classification_pipeline_uses_created_classes():
    pipelines = build_pipeline_graph(
        input_types=(Seq[AudioFile], Supervised[VectorCategorical]),
        output_type=VectorCategorical,
        registry=find_classes("Keras")
        + [AudioClassifier, AudioCommandPreprocessor, KerasAudioClassifier],
    )
    nodes = pipelines.nodes()
    assert AudioClassifier in nodes

    pipelines = build_pipeline_graph(
        input_types=AudioFile,
        output_type=AudioFeatures,
        registry=find_classes("Keras")
        + [AudioClassifier, AudioCommandPreprocessor, KerasAudioClassifier],
    )
    nodes = pipelines.nodes()
    assert AudioCommandPreprocessor in nodes

    pipelines = build_pipeline_graph(
        input_types=AudioFile,
        output_type=AudioCommand,
        registry=find_classes("Keras")
        + [
            AudioClassifier,
            AudioCommandPreprocessor,
            KerasAudioClassifier,
            AudioCommandReader,
        ],
    )
    nodes = pipelines.nodes()
    assert AudioCommandReader in nodes


def test_algorithm_correct_types():
    assert AudioClassifier.input_types() == (
        Seq[AudioFile],
        Supervised[VectorCategorical],
    )

    assert AudioCommandPreprocessor.input_types() == (AudioFile,)
    assert AudioCommandPreprocessor.output_type() == AudioFeatures

    assert AudioCommandReader.input_types() == (AudioFile,)
    assert AudioCommandReader.output_type() == AudioCommand
