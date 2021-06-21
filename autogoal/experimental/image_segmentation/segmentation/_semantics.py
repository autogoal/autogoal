from autogoal.kb._semantics import Tensor3, Tensor, SemanticType
from os import path


Image = Tensor3
ImageMask = Tensor

class ImageFile(SemanticType):
    """Semantic type for JPG image files"""

    @classmethod
    def _match(cls, x):
        try:
            return path.isfile(x) and x.split(".")[-1] in ["jpg", "png"]
        except TypeError:
            return False
