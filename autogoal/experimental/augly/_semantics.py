from autogoal.kb._semantics import SemanticType
from os import path

class Image(SemanticType):
    """Semantic type for JPG image files"""

    @classmethod
    def _match(cls, x):
        try:
            return path.isfile(x) and x.split(".")[-1] in ["jpg"]
        except TypeError:
            return False