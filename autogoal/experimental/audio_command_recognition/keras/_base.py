import numpy as np
from autogoal.utils import nice_repr
from autogoal.contrib.keras._base import KerasClassifier
from autogoal.contrib.keras._grammars import generate_grammar
from ._grammars import Modules
from autogoal.experimental.audio_command_recognition.kb._semantics import AudioFeatures
from autogoal.kb import Seq, Supervised, VectorCategorical
from tensorflow.keras.layers import Input


@nice_repr
class KerasAudioClassifier(KerasClassifier):
    def __init__(self, **kwargs):
        super().__init__(optimizer="adam", **kwargs)

    def _build_grammar(self):
        return generate_grammar(Modules.Conv1D())

    def run(
        self, X: Seq[AudioFeatures], y: Supervised[VectorCategorical]
    ) -> VectorCategorical:
        X = np.array(X)
        return super().run(X, y)

    def _build_input(self, X):
        X = np.array(X)
        return Input(shape=X.shape[1:])
