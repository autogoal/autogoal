import pprint
import json

from autogoal.ml import DatasetFeatureExtractor
from autogoal.datasets import (
    abalone,
    cars,
    cifar10,
    dorothea,
    german_credit,
    gisette,
    haha,
    meddocan,
    movie_reviews,
    shuttle,
    wine_quality,
    yeast,
)


from autogoal.ml import (
    AutoML,
    DatasetFeatureLogger,
    LearnerMedia,
    DatasetFeatureExtractor,
)
from autogoal.ml._metalearning import SolutionInfo
from autogoal.contrib.keras import KerasClassifier
from autogoal.kb import List, Word, Postag, CategoricalVector, Sentence, Tensor4
from autogoal.contrib import find_classes


def run_automl(X, y, input=None, output=None):
    automl = AutoML(
        search_iterations=1000,
        metalearning_log=True,
        search_kwargs=dict(search_timeout=2 * 60 * 60,),
        errors="ignore",
        input=input,
        output=output,
        cross_validation_steps=1,
    )


for dataset in [
    abalone,
    cars,
    dorothea,
    german_credit,
    gisette,
    shuttle,
    wine_quality,
    yeast,
]:
    X, y, *_ = dataset.load()
    run_automl(X, y)


for dataset in [movie_reviews, haha]:
    X, y, *_ = dataset.load()
    run_automl(X, y, input=List(Sentence()), output=CategoricalVector())


for dataset in [meddocan]:
    X, _, y, _ = dataset.load()
    run_automl(X, y, input=List(List(Word())), output=List(List(Postag())))


for dataset in [cifar10]:
    X, y, *_ = dataset.load()
    run_automl(X, y, input=Tensor4(), output=CategoricalVector())
