from autogoal.kb._semantics import Tensor3, Tensor, SemanticType, Discrete, Dense
from os import path


Image = Tensor3
ImageMask = Tensor[2, Discrete, Dense]

class ImageFile(SemanticType):
    """Semantic type for JPG and PNG image files"""

    @classmethod
    def _match(cls, x):
        try:
            return path.isfile(x) and x.split(".")[-1] in ["jpg", "png"]
        except TypeError:
            return False
