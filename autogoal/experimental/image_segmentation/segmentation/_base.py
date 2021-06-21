from autogoal.utils import nice_repr
from autogoal.kb._algorithm import Supervised, AlgorithmBase, algorithm, Seq


import tensorflow as tf
import numpy as np

from ._semantics import Image, ImageMask, ImageFile
from ..kb._generated import ImageReader


@nice_repr
class ImageSegmenter(AlgorithmBase):
    """
    Class for a generic image segmenter, receives images and returns segmentation masks with same size
    """

    def __init__(self, segmenter: algorithm(Seq[Image], Supervised[Seq[ImageMask]], Seq[ImageMask]),
                 image_preprocessor: algorithm(ImageFile, Image)):
        self._segmenter = segmenter
        self._mode = "train"
        self.image_preprocessor = image_preprocessor

    def train(self):
        self._mode = "train"
        self._segmenter.train()

    def eval(self):
        self._mode = "eval"
        self._segmenter.eval()

    def fit(self, images, masks):
        self._segmenter.fit(self._preprocess_images(images), self._preprocess_masks(masks))

    def _preprocess_images(self, images) -> Seq[Image]:
        p_images = []
        for image in images:
            p_images.append(self.image_preprocessor.run(image))
        return np.array(p_images)

    def _preprocess_masks(self, images) -> Seq[ImageMask]:
        reader = ImageReader()
        p_images = []
        for image_file in images:
            t = reader.run(image_file)
            resize = tf.image.resize(t, [512, 512])
            resize = resize[:, :, 0]
            p_images.append(np.array(resize) - 1)
        return np.array(p_images)

    def predict(self, images):
        return self._segmenter.predict(self._preprocess_images(images))

    def run(self, data: Seq[ImageFile], masks: Supervised[Seq[ImageMask]]) -> Seq[ImageMask]:
        if self._mode == "train":
            self.fit(data, masks)
            return masks
        if self._mode == "eval":
            return self.predict(data)


@nice_repr
class ImagePreprocessor(AlgorithmBase):
    """
    Receives image file and properly preprocesses it 
    """

    def __init__(self) -> None:
        self.reader = ImageReader()

    def run(self, image_file: ImageFile) -> Image:
        t = self.reader.run(image_file)
        resize = tf.image.resize(t, [512, 512])
        return np.array(resize)
