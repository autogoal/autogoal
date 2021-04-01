import io
import pickle
import statistics

import numpy as np
from autogoal.contrib import find_classes

from autogoal.kb import build_pipeline_graph, SemanticType
from autogoal.ml.metrics import accuracy
from autogoal.search import PESearch
from autogoal.utils import nice_repr


@nice_repr
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
        search_algorithm=None,
        search_iterations=100,
        include_filter=".*",
        exclude_filter=None,
        validation_split=0.3,
        errors="warn",
        cross_validation="median",
        cross_validation_steps=3,
        registry=None,
        score_metric=None,
        **search_kwargs
    ):
        self.input = input
        self.output = output
        self.search_algorithm = search_algorithm or PESearch
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
        self.search_kwargs = search_kwargs
        self._unpickled = False

        if random_state:
            np.random.seed(random_state)

    def _check_fitted(self):
        if not hasattr(self, "best_pipeline_"):
            raise TypeError(
                "This operation cannot be performed on an unfitted AutoML instance. Call `fit` first."
            )

    def make_pipeline_builder(self):
        registry = self.registry or find_classes(
            include=self.include_filter, exclude=self.exclude_filter
        )

        return build_pipeline_graph(
            input_types=self.input, output_type=self.output, registry=registry,
        )

    def fit(self, X, y, **kwargs):
        self.input = self._input_type(X)
        self.output = self._output_type(y)

        search = self.search_algorithm(
            self.make_pipeline_builder(),
            self.make_fitness_fn(X, y),
            random_state=self.random_state,
            errors=self.errors,
            **self.search_kwargs,
        )

        self.best_pipeline_, self.best_score_ = search.run(
            self.search_iterations, **kwargs
        )

        self.fit_pipeline(X, y)

    def fit_pipeline(self, X, y):
        self._check_fitted()

        self.best_pipeline_.send("train")
        self.best_pipeline_.run(X, y)
        self.best_pipeline_.send("eval")

    def save(self, fp: io.BytesIO):
        """
        Serializes the AutoML instance.
        """
        self._check_fitted()
        pickle.Pickler(fp).dump(self)

    @classmethod
    def load(self, fp: io.FileIO) -> "AutoML":
        """
        Deserializes an AutoML instance. 
        
        After deserialization, the best pipeline found is ready to predict.
        """
        automl = pickle.Unpickler(fp).load()

        if not isinstance(automl, AutoML):
            raise ValueError("The serialized file does not contain an AutoML instance.")

        return automl

    def score(self, X, y):
        self._check_fitted()

        y_pred = self.best_pipeline_.run(X, np.zeros_like(y))
        return self.score_metric(y, y_pred)

    def _input_type(self, X):
        return self.input or SemanticType.infer(X)

    def _output_type(self, y):
        return self.output or SemanticType.infer(y)

    def make_fitness_fn(self, X, y):
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
                pipeline.run(X_train, y_train)
                pipeline.send("eval")
                y_pred = pipeline.run(X_test, None)
                scores.append(self.score_metric(y_test, y_pred))

            return getattr(statistics, self.cross_validation)(scores)

        return fitness_fn

    def predict(self, X):
        self._check_fitted()

        return self.best_pipeline_.run(X, None)
