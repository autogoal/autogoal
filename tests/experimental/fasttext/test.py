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

#X_train, y_train, X_test, y_test = haha.load()
#X_train, y_train, X_test, y_test =["","",""],array([1,2,3]),["","",""],array([1,2,3])
X_train = []
y_train = []
X_test = []
y_test = []

with open('cooking.train',"r") as f:
    lines = []
    for line in f:
        labels = re.findall("__label__[a-z,A-Z,-]+",line)
        text = re.split("__label__[a-z,A-Z,-]+ ", line)[-1]
        X_train.append(text)
        y_train.append(Text_Descriptor( *labels))

with open('cooking.valid',"r") as f:
    lines = []
    for line in f:
        labels = re.findall("__label__[a-z,A-Z,-]+",line)
        text = re.split("__label__[a-z,A-Z,-]+ ", line)[-1]
        X_test.append(text)
        y_test.append(Text_Descriptor( *labels))


automl = AutoML(
    input=(Seq[Sentence], Supervised[VectorCategorical]),  # **tipos de entrada**
    output= VectorCategorical,  # **tipo de salida**
    registry= [SupervisedTextClassifier],#+find_classes() ,
    evaluation_timeout= 5 * Min,
    memory_limit=3.5 * Gb,
    search_timeout= 2 * Min,
    #errors="raise"
)
# a = SupervisedTextClassifier(1,25,1.0,2)
# a.run(X_train , y_train)
# print(a._eval(["Which baking dish is best to bake a banana bread ?","__label__oven How does grill/broil mode in a convection oven work?"]))
from autogoal.search import RichLogger
#automl.fit(X_train, y_train, logger=RichLogger())
automl.fit(X_train,y_train,logger=RichLogger())
a = automl.best_pipeline_
#print(a.epoch ,a.lr , a.wordNgrams )
score = automl.score(X_test, y_test)
print(score)
