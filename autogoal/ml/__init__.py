from autogoal.search import RandomSearch, PESearch
from autogoal.kb import (
    build_pipelines,
    CategoricalVector,
    List,
    Word,
    MatrixContinuous,
    Tuple,
    infer_type,
    Postag,
    Chunktag
)

from autogoal.ml.metrics import accuracy

from autogoal.contrib import find_classes
import numpy as np
import random
import statistics


class AutoML:
    """
    Predefined pipeline search with automatic type inference.

    An `AutoML` instance represents a general-purpose machine learning
    algorithm, that can be applied to any input and output.
    """
    def __init__(
        self,
        input=None,
        output=None,
        random_state=None,
        search_algorithm=PESearch,
        search_kwargs={},
        search_iterations=100,
        include_filter=".*",
        exclude_filter=None,
        validation_split=0.3,
        errors="warn",
        cross_validation="median",
        cross_validation_steps=3,
        registry=None,
        score_metric=None,
    ):
        self.input = input
        self.output = output
        self.search_algorithm = search_algorithm
        self.search_kwargs = search_kwargs
        self.search_iterations = search_iterations
        self.include_filter = include_filter
        self.exclude_filter = exclude_filter
        self.validation_split = validation_split
        self.errors = errors
        self.cross_validation = cross_validation
        self.cross_validation_steps = cross_validation_steps
        self.registry = registry
        self.random_state = random_state
        self.score_metric = score_metric or accuracy

        if random_state:
            np.random.seed(random_state)

    def fit(self, X, y, **kwargs):
        registry = self.registry or find_classes(
            include=self.include_filter, exclude=self.exclude_filter
        )

        output_type = self._output_type(y)

        self.pipeline_builder_ = build_pipelines(
            input=Tuple(self._input_type(X), output_type),
            output=output_type,
            registry=registry,
        )

        search = self.search_algorithm(
            self.pipeline_builder_,
            self._make_fitness_fn(X, y),
            random_state=self.random_state,
            errors=self.errors,
            **self.search_kwargs,
        )

        self.best_pipeline_, self.best_score_ = search.run(
            self.search_iterations, **kwargs
        )

        self.best_pipeline_.send("train")
        self.best_pipeline_.run((X, y))
        self.best_pipeline_.send("eval")

    def score(self, X, y):
        y_pred = self.best_pipeline_.run((X, np.zeros_like(y)))
        return self.score_metric(y, y_pred)

    def _input_type(self, X):
        return self.input or infer_type(X)

    def _output_type(self, y):
        return self.output or infer_type(y)

    def _make_fitness_fn(self, X, y):
        if isinstance(X, list):
            X = np.asarray(X)

        y = np.asarray(y)

        def fitness_fn(pipeline):
            scores = []

            for _ in range(self.cross_validation_steps):
                indices = np.arange(0, X.shape[0])
                np.random.shuffle(indices)
                split_index = int(self.validation_split * len(indices))
                train_indices = indices[:-split_index]
                test_indices = indices[-split_index:]

                X_train, y_train, X_test, y_test = (
                    X[train_indices],
                    y[train_indices],
                    X[test_indices],
                    y[test_indices],
                )

                pipeline.send("train")
                pipeline.run((X_train, y_train))
                pipeline.send("eval")
                y_pred = pipeline.run((X_test, np.zeros_like(y_test)))
                scores.append(self.score_metric(y_test, y_pred))

            return getattr(statistics, self.cross_validation)(scores)

        return fitness_fn

    def predict(self, X):
        return self.best_pipeline_.run((X, [None] * len(X)))
