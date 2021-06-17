from autogoal.utils import Min
from autogoal.ml import AutoML
from autogoal.kb import *
from ._semantics import Image
from autogoal.contrib import find_classes
from ..keras._base import KerasImageSegmenter
from ._base import ImageSegmenter




def run_example():
    automl= AutoML(input=(Seq[Image], Seq[Image]), output=Image, cross_validation_steps=1, registry=find_classes()+[ImageSegmenter, KerasImageSegmenter])
    
    
     