import imp
from threading import TIMEOUT_MAX
from numpy.core.records import array
import autogoal
from enum import auto
from autogoal import contrib
from autogoal.kb import AlgorithmBase

from autogoal.grammar import BooleanValue, DiscreteValue
from autogoal.kb import *
from autogoal.ml import AutoML
from autogoal.contrib import find_classes
from autogoal.experimental.fasttex._base import  SupervisedTextClassifier , Text_Descriptor
from autogoal.contrib.spacy import SpacyNLP
from autogoal.datasets import haha
from autogoal.utils import Min, Gb
import numpy as np
import re


X_train, y_train , X_test , y_test = load()
            

automl = AutoML(
    input=(Seq[Sentence], Supervised[VectorCategorical]),  # **tipos de entrada**
    output= VectorCategorical,  # **tipo de salida**    
    registry= [SupervisedTextClassifier]+find_classes() ,
    evaluation_timeout= 30 * Min,
    memory_limit=3.5 * Gb,
    search_timeout= 2 * Min,
    #errors="raise"
)

from autogoal.search import RichLogger
automl.fit(X_train,y_train,logger=RichLogger())
score = automl.score(X_test, y_test)
print(score)
