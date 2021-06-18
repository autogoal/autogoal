from autogoal.ml import AutoML
from autogoal.kb import *
from ._semantics import Image
from autogoal.contrib import find_classes
from ..keras._base import KerasImageSegmenter
from ._base import ImageSegmenter, ImagePreprocessor
from ..data.dataset import load
from autogoal.search import RichLogger


def test():
    automl = AutoML(input=(Seq[Image], Seq[Image]), output=Seq[Image], cross_validation_steps=1,
                    registry=find_classes() + [ImageSegmenter, KerasImageSegmenter, ImagePreprocessor])
    
    x_train, y_train, x_test, y_test = load()

    automl.fit(x_train, y_train, logger=[RichLogger()])
    score = automl.score(x_test, y_test)
    print(score)