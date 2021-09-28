from autogoal.kb._semantics import SemanticType
from numpy import ndarray


class Audio(SemanticType):
    """
    Define a semantic type for audios or sounds, currently support `numpy.ndarray`
    """

    @classmethod
    def _match(cls, x) -> bool:
        return isinstance(x, ndarray)
