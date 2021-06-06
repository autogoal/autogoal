from autogoal.ml import AutoML
from autogoal.kb import *
from autogoal.search import RichLogger
from autogoal.utils import Min, Gb
from autogoal.experimental.audio_command_recognition.kb._semantics import AudioFile
from autogoal.experimental.audio_command_recognition.audio._base import (
    AudioClassifier,
    AudioCommandPreprocessor,
)
from autogoal.experimental.audio_command_recognition.keras._base import (
    KerasAudioClassifier,
)
from autogoal.experimental.audio_command_recognition.datasets.audio_commands import load
from autogoal.contrib import find_classes
import numpy as np


def run_example():
    automl = AutoML(
        input=(Seq[AudioFile], Supervised[VectorCategorical]),
        output=VectorCategorical,
        cross_validation_steps=1,
        registry=find_classes()
        + [AudioClassifier, AudioCommandPreprocessor, KerasAudioClassifier],
        evaluation_timeout=10 * Min,
        memory_limit=3.5 * Gb,
        search_timeout=30 * Min,
    )

    x_train, y_train, x_test, y_test = load()

    automl.fit(x_train, y_train, logger=[RichLogger()])
    score = automl.score(x_test, y_test)
    print(score)

