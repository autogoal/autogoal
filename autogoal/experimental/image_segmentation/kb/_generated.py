from tensorflow.python.ops.gen_io_ops import read_file
from autogoal.kb import AlgorithmBase
from autogoal.utils import nice_repr
from ._semantics import ImageFile
from ..segmentation._semantics import Image, ImageMask
from tensorflow import io
# from os import open, read
# from skimage import filters

@nice_repr
class ImageReader(AlgorithmBase):
    """
    Reader of image files.
    """
    def run(self, image_file: ImageFile) -> Image:
        file = open(image_file, 'rb')
        image = io.decode_image(file.read())
        return image

@nice_repr
class ImageMaskReader(AlgorithmBase):
    """
    Reader of imageMask files.
    """
    def run(self, image_file: ImageFile) -> ImageMask:
        file = open(image_file, 'rb')
        image = io.decode_image(file.read())
        return image

# @nice_repr
# class GaussianFilter:
#     """
#     Denoises image using gaussian filter
#     """
#
#     def run(self, image: Image) -> Image:
#         return filters.gaussian(image, sigma=2)
