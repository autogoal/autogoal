from os import path
from autogoal.kb import SemanticType

class AudioFile(SemanticType):
    """Semantic type for wav audio files"""

    @classmethod
    def _match(cls, x):
        try:
            return path.isfile(x) and x.split(".")[-1] == "wav"
        except TypeError:
            return False
