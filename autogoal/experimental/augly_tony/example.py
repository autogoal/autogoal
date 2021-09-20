from autogoal.kb import AlgorithmBase

from autogoal.grammar import BooleanValue, DiscreteValue
from autogoal.kb import *

from autogoal.ml import AutoML
from autogoal.contrib import find_classes

from autogoal.datasets import haha

from transformers import (
    ReplaceSimilarCharsTransformer,
    InsertPunctuationCharsTransformer,
    SplitWordsTransformer,
)

X_train, y_train, X_test, y_test = haha.load()

automl = AutoML(
    input=(Seq[Sentence], Supervised[VectorCategorical]),  # **tipos de entrada**
    output=VectorCategorical,
    registry=[
        ReplaceSimilarCharsTransformer,
        InsertPunctuationCharsTransformer,
        SplitWordsTransformer,
    ]
    + find_classes(),
)

automl.fit(X_train, y_train)

score = automl.score(X_test, y_test)
print(score)
