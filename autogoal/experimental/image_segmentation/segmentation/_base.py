from autogoal.utils import nice_repr
from autogoal.kb._algorithm import Supervised
from autogoal.kb import AlgorithmBase, algorithm, Seq
from ._semantics import Image
import numpy as np
from ..kb._generated import ImageReader, ImageMaskReader
from ..kb._semantics import ImageFile
from ..segmentation._semantics import Image, ImageMask



@nice_repr
class ImageSegmenter(AlgorithmBase):
    """
    Receives images and returns segmentation masks with same size
    """

    def __init__(self, segmenter: algorithm(Seq[Image], Supervised[Seq[ImageMask]], Seq[ImageMask]), image_preprocessor: algorithm(ImageFile, Image), mask_preprocessor: algorithm(ImageFile, ImageMask)):
        self._segmenter = segmenter
        self._mode = "train"
        self.image_preprocessor=image_preprocessor
        self.mask_preprocessor=mask_preprocessor

    def train(self):
        self._mode = "train"
        self._segmenter.train()

    def eval(self):
        self._mode = "eval"
        self._segmenter.eval()

    def fit(self, images, masks):
        self._segmenter.fit(self._preprocess_images(images), self._preprocess_masks(masks))
    
    def _preprocess_images(self, images):
        p_images=[]
        for image in images:
            p_images.append(self.image_preprocessor.run(image))
            
        return np.array(p_images)
    
    def _preprocess_masks(self, images):
        p_images=[]
        for image in images:
            p_images.append(self.mask_preprocessor.run(image))
            
        return np.array(p_images)
        

    def predict(self, images):
        return self._segmenter.predict(self._preprocess_images(images))

    def run(self, data: Seq[Image], masks: Seq[ImageMask]) -> Seq[ImageMask]:
        if self._mode == "train":
            self.fit(data, masks)
            return masks
        if self._mode == "eval":
            return self.predict(data)

@nice_repr
class ImagePreprocessor(AlgorithmBase):
    """
    Receives image file and converts it into appropriate input
    """
    def __init__(self) -> None:
        self.reader=ImageReader()
        
    def run(self, image_file: ImageFile) -> Image:
        return self.reader.run(image_file)
    
@nice_repr
class ImageMaskPreprocessor(AlgorithmBase):
    """
    Receives image mask file and converts it into appropriate input
    """
    def __init__(self) -> None:
        self.reader=ImageMaskReader()
        
    def run(self, image_file: ImageFile) -> Image:
        return self.reader.run(image_file)