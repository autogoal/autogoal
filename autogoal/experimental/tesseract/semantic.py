from os import path
from autogoal.kb import SemanticType


class Image_File(SemanticType):
    """Semantic type for images files"""

    @classmethod
    def _match(cls, x):
        supported_formats = ["jpeg", "jpg", "png", "gif", "bmp", "tiff", "webp", "pbm", "pgm", "ppm"]
                
        try:
            return path.isfile(x) and x.split(".")[-1] in supported_formats
        except TypeError:
            return False