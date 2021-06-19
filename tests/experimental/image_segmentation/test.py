from autogoal.ml import AutoML
from autogoal.kb import *
from autogoal.experimental.image_segmentation.kb._semantics import ImageFile
from autogoal.experimental.image_segmentation.segmentation._semantics import Image
from autogoal.contrib import find_classes
from autogoal.experimental.image_segmentation.segmentation._base import ImageSegmenter, ImagePreprocessor
from autogoal.experimental.image_segmentation.keras._base import KerasImageSegmenter
from autogoal.experimental.image_segmentation.data.dataset import load
from autogoal.search import RichLogger


def test():
    automl = AutoML(
        input=(Seq[ImageFile], Supervised[Seq[Image]]),
        output=Seq[Image],
        registry=find_classes() + [ImagePreprocessor, ImageSegmenter, KerasImageSegmenter]
    )

    x_train, y_train, x_test, y_test = load()
    automl.fit(x_train, y_train, logger=[RichLogger()])
    score = automl.score(x_test, y_test)
    print(score)


test()
