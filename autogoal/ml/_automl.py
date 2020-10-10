
from autogoal.search import RandomSearch, PESearch
from autogoal.kb import (
    build_pipeline_graph,
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
from autogoal.sampling import ReplaySampler
from autogoal.contrib import find_classes
# from autogoal.ml._metalearning import DatasetFeatureLogger

import numpy as np
import random
import statistics
import pickle



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
        metalearning_log=False,
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
        self.metalearning_log = metalearning_log

        if random_state:
            np.random.seed(random_state)

    def _make_pipeline_builder(self):
        registry = self.registry or find_classes(
            include=self.include_filter, exclude=self.exclude_filter
        )

        return build_pipelines(
            input=Tuple(self.input, self.output),
            output=self.output,
            registry=registry,
        )

    def fit(self, X, y, **kwargs):
        self.input = self._input_type(X)
        self.output = self._output_type(y)

        if self.metalearning_log:
            raise NotImplementedError("Metalearning is not ready yet")

            loggers = kwargs.get('logger', [])
            loggers.append(DatasetFeatureLogger(X, y, problem_features=dict(
                input=repr(self.input),
                output=repr(self.output),
                metric=self.score_metric.__name__,
            ), environment_features=dict(
                memory_limit=self.search_kwargs.get('memory_limit'),
                search_timeout=self.search_kwargs.get('search_timeout'),
                evaluation_timeout=self.search_kwargs.get('evaluation_timeout'),
            )))
            kwargs['logger'] = loggers

        search = self.search_algorithm(
            self._make_pipeline_builder(),
            self._make_fitness_fn(X, y),
            random_state=self.random_state,
            errors=self.errors,
            **self.search_kwargs,
        )

        self.best_pipeline_, self.best_score_ = search.run(
            self.search_iterations, **kwargs
        )

        self.fit_pipeline(X, y)

    def fit_pipeline(self, X, y):
        if not hasattr(self, 'best_pipeline_'):
            raise TypeError("You have to call `fit()` first.")

        self.best_pipeline_.send("train")
        self.best_pipeline_.run((X, y))
        self.best_pipeline_.send("eval")

    def save_pipeline(self, fp):
        """
        Saves the state of the best pipeline.
        You are responsible for opening and closing the stream.
        """
        if not hasattr(self, 'best_pipeline_'):
            raise TypeError("You have to call `fit()` first.")

        self.best_pipeline_.sampler_.replay().save(fp)
        pickle.Pickler(fp).dump((self.input, self.output))

    def load_pipeline(self, fp):
        """
        Loads the state of the best pipeline and retrains.
        You are responsible for opening and closing the stream.

        After calling load, the best pipeline is **not** trained.
        You need to retrain it by calling `fit_pipeline(X, y)`.
        """
        sampler = ReplaySampler.load(fp)
        self.input, self.output = pickle.Unpickler(fp).load()
        self.best_pipeline_ = self._make_pipeline_builder()(sampler)

    def score(self, X, y):
        y_pred = self.best_pipeline_.run((X, np.zeros_like(y)))
        return self.score_metric(y, y_pred)

    def _input_type(self, X):
        return self.input or infer_type(X)

    def _output_type(self, y):
        return self.output or infer_type(y)

    def _make_fitness_fn(self, X, y):
        y = np.asarray(y)

        def fitness_fn(pipeline):
            scores = []

            for _ in range(self.cross_validation_steps):
                len_x = len(X) if isinstance(X, list) else X.shape[0]
                indices = np.arange(0, len_x)
                np.random.shuffle(indices)
                split_index = int(self.validation_split * len(indices))
                train_indices = indices[:-split_index]
                test_indices = indices[-split_index:]

                if isinstance(X, list):
                    X_train, y_train, X_test, y_test = (
                        [X[i] for i in train_indices],
                        y[train_indices],
                        [X[i] for i in test_indices],
                        y[test_indices],
                    )
                else:
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
