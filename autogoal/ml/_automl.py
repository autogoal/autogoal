import io
import pathlib
import pickle
import statistics
import os
import shutil

import numpy as np
from autogoal.contrib import find_classes
from pathlib import Path

from autogoal.kb import build_pipeline_graph, SemanticType, Pipeline
from autogoal.ml.metrics import accuracy
from autogoal.search import PESearch
from autogoal.utils import nice_repr, generate_production_dockerfile


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
        **search_kwargs,
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
            input_types=self.input,
            output_type=self.output,
            registry=registry,
        )

    def fit(self, X, y=None, **kwargs):
        self.input = self._input_type(X)

        if not y is None:
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

        for pipeline in self.best_pipeline_:
            pipeline.send("train")
            pipeline.run(X, y)
            pipeline.send("eval")

    def save(self, fp: io.BytesIO):
        """
        Serializes the AutoML instance.
        """
        self._check_fitted()
        pickle.Pickler(fp).dump(self)

    def folder_save(self, path: Path):
        """
        Serializes the AutoML into a given path.
        """
        if path is None:
            path = os.getcwd()

        self._check_fitted()
        save_path = path / "storage"

        try:
            os.makedirs(save_path)
        except:
            shutil.rmtree(save_path)
            os.makedirs(save_path)

        for i, pipeline in enumerate(self.best_pipeline_):
            solution_path = save_path / f"solution_{i}"
            tmp = pipeline.algorithms
            pipeline.save_algorithms(solution_path)
            pipeline.algorithms = []
            with open(solution_path / "model.bin", "wb") as fd:
                self.save(fd)
            pipeline.algorithms = tmp

        generate_production_dockerfile(path)

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

    @classmethod
    def folder_load(self, path: Path) -> "AutoML":
        """
        Deserializes an AutoML instance from a given path.

        After deserialization, the best pipeline found is ready to predict.
        """
        load_path = path / "storage"
        with open(load_path / "model.bin", "rb") as fd:
            automl = self.load(fd)

        pipelines = []
        for i in [1, 2, 3]:
            solution_path = load_path / f"solution_{i}"
            pipelines.append(Pipeline.load_algorithms(solution_path))

        automl.best_pipeline_ = pipelines
        return automl

    def score(self, X, y):
        self._check_fitted()

        scores = []
        for pipeline in self.best_pipeline_:
            y_pred = pipeline.run(X, np.zeros_like(y))
            scores.append(self.score_metric(y, y_pred))
        return scores

    def _input_type(self, X):
        return self.input or SemanticType.infer(X)

    def _output_type(self, y):
        return self.output or SemanticType.infer(y)

    def make_fitness_fn(self, X, y=None):
        if not y is None:
            y = np.asarray(y)

        def fitness_fn(pipeline):
            return self.score_metric(
                pipeline,
                X,
                y,
                self.validation_split,
                self.cross_validation_steps,
                self.cross_validation,
            )

        return fitness_fn

    def predict_all(self, X):
        self._check_fitted()

        return [pipeline.run(X, None) for pipeline in self.best_pipeline_]

    def predict(self, solution_index, X):
        self._check_fitted()

        return self.best_pipeline_[solution_index].run(X, None)

    def export(self, name):
        """
        Exports the result of the AutoML run onto a new Docker image.
        """
        self.save()
        os.system(
            "docker build --file ./dockerfiles/production/dockerfile-safe -t autogoal:production ."
        )
        os.system(f"docker save -o {name}.tar autogoal:production")

    def export_portable(self, path=None):
        """
        Generate a portable set of files that can be used to export the model into a new Docker image
        """
        if path is None:
            path = os.getcwd()

        datapath = f"{path}/autogoal-export"
        final_path = Path(datapath)
        if final_path.exists():
            shutil.rmtree(datapath)

        self.folder_save(final_path)

        makefile = open(final_path / "makefile", "w")
        makefile.write(
            """
build:

	docker build --file ./dockerfile -t autogoal:production .
    docker save -o autogoal-prod.tar autogoal:production
        """
        )
        makefile.close()

        print("generated assets for production deployment")
