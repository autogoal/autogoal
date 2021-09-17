from autogoal.kb._semantics import SemanticType
from PIL.Image import Image as PILImage

class Image(SemanticType):
    '''
    Define a semantic type for images, from `PIL.Image.Image`
    '''

    @classmethod
    def _match(cls, x) -> bool:
        return isinstance(x, PILImage)