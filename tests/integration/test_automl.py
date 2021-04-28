from enum import auto
from autogoal.ml import AutoML
from autogoal.kb import (
    MatrixContinuousDense,
    VectorCategorical,
    Supervised,
    Seq,
    Sentence,
)

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


def run_automl_test(
    dataset, input, output, search_timeout, evaluation_timeout, expected_fitness
):
    automl = AutoML(
        input=input,
        output=output,
        evaluation_timeout=evaluation_timeout,
        search_timeout=search_timeout,
        cross_validation_steps=1,
    )

    X, y = dataset
    automl.fit(X, y)

    assert automl.best_score_ >= expected_fitness


def test_cars():
    X, y = cars.load()

    run_automl_test(
        dataset=(X, y),
        input=(MatrixContinuousDense, Supervised[VectorCategorical]),
        output=VectorCategorical,
        search_timeout=60,
        evaluation_timeout=5,
        expected_fitness=0.9,
    )


def test_movie_reviews():
    X, y = movie_reviews.load(max_examples=100)

    run_automl_test(
        dataset=(X, y),
        input=(Seq[Sentence], Supervised[VectorCategorical]),
        output=VectorCategorical,
        search_timeout=60,
        evaluation_timeout=5,
        expected_fitness=0.6,
    )
