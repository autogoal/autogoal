import pprint
import json

from autogoal.ml import DatasetFeatureExtractor
from autogoal.datasets import cars, abalone, german_credit, cifar10, haha, meddocan


# for dataset in [cars, abalone, german_credit, cifar10, haha, meddocan]:
#     X, y, *_ = dataset.load()
#     features = DatasetFeatureExtractor().extract_features(X, y)
#     pprint.pprint(dict(dataset=dataset.__name__, features=features))


# Testing that AutoML saves model

from autogoal.ml import (
    AutoML,
    DatasetFeatureLogger,
    LearnerMedia,
    DatasetFeatureExtractor,
)
from autogoal.ml._metalearning import SolutionInfo
from autogoal.contrib.keras import KerasClassifier
from autogoal.kb import List, Word, Postag, CategoricalVector, Sentence
from autogoal.contrib import find_classes


# X, y = cars.load()
# X, _, y, _ = meddocan.load(max_examples=100)
X, _, y, _ = haha.load(max_examples=100)


automl = AutoML(
    search_iterations=1,
    metalearning_log=True,
    # input=List(List(Word())),
    # output=List(List(Postag())),
    input=List(Sentence()),
    output=CategoricalVector(),

    registry=find_classes(exclude=".*Keras.*")
)
automl.fit(X, y)
