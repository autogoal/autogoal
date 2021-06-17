from autogoal.utils import nice_repr
from autogoal.kb._algorithm import Supervised
from autogoal.kb import AlgorithmBase, algorithm, Seq
from ._semantics import Image
import numpy as np


@nice_repr
class ImageSegmenter(AlgorithmBase):
    """
    Receives images and returns segmentation masks with same size
    """

    def __init__(self, segmenter: algorithm(Seq[Image], Supervised[Seq[Image]], Image)):
        self._segmenter = segmenter
        self._mode = "train"

    def train(self):
        self._mode = "train"
        self._segmenter.train()

    def eval(self):
        self._mode = "eval"
        self._segmenter.eval()

    def fit(self, images, masks):
        images = np.array(images)
        self._segmenter.fit(images, masks)

    def predict(self, images):
        images = np.array(images)
        return self._segmenter.predict(images)

    def run(self, data: Seq[Image], masks: Seq[Image]) -> Image:
        if self._mode == "train":
            self.fit(data, masks)
            return masks
        if self._mode == "eval":
            return self.predict(data)
