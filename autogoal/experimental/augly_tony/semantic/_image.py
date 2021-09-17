from autogoal.kb._semantics import SemanticType
from PIL.Image import Image as img
from numpy import ndarray
from os import path
import imghdr

class Image(SemanticType):
    '''
    Define a semantic type for images, currently support:
    
    • path to the image as `str`

    • `Image` from `PIL.Image`

    • `ndarray` from `numpy`
    '''


    @classmethod
    def _match(cls, x) -> bool:
        try:
            if isinstance(x,str):
                if not path.isfile(x):
                    return False
                return imghdr.what(path.split[1]) is not None   # confirm that this file is actually an image
        except TypeError:
            return False

        return isinstance(x, img) or isinstance(x, ndarray) or super()._match(x)
       
   
