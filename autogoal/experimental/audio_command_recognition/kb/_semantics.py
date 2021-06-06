from autogoal.kb._semantics import SemanticType, Tensor, Continuous, Dense
from os import path

# Semantics types for audio processing
class AudioFile(SemanticType):
    """Semantic type for wav audio files"""

    @classmethod
    def _match(cls, x):
        try:
            return path.isfile(x) and x.split(".")[-1] == "wav"
        except TypeError:
            return False


class AudioCommand(SemanticType):
    """Semantic type for vectors representing audio commands"""

    @classmethod
    def _match(cls, x):
        pass


AudioFeatures = Tensor[3, Continuous, Dense]
