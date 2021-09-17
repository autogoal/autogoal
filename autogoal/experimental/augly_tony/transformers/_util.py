from autogoal.kb import AlgorithmBase

from typing import Tuple
from PIL.Image import Image as PLImage
from augly.image.transforms import BaseTransform

from augly_tony.semantic import Image

class AugLyTransformer(AlgorithmBase):
    def __init__(self):
        super().__init__()
        self._transformer: BaseTransform = None

    def get_transformer(self) -> BaseTransform:
        pass
   


def discrete_to_color(color: int) -> Tuple[int, int, int]:
    '''
    Convert a 8bit color `int` representation to `(r,g,b)` representation
    '''
    return (
        # Convert discrete value to a color Tuple using a mask
        color & 0xFF0000,
        color & 0x00FF00,
        color & 0x0000FF,
    )

def image_to_PIL_image(image: Image) -> PLImage:
    
    assert(Image._match(image), "Can't not convert to image")

    

