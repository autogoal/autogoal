from tensorflow.python.ops.gen_io_ops import read_file
from autogoal.kb import AlgorithmBase
from autogoal.utils import nice_repr
from ._semantics import ImageFile
from ..segmentation._semantics import Image
from tensorflow import io
from os import open, read
from skimage import filters

@nice_repr
class ImageReader(AlgorithmBase):
    """
    Reader of image files.
    """
    def run(self, image_file: ImageFile) -> Image:
        return io.decode_image(read(open(image_file, 'r')))
        
        
@nice_repr
class GaussianFilter:
    """
    Denoises image using gaussian filter 
    """
    
    def run(self, image: Image) -> Image:
        return filters.gaussian(image, sigma=2)
