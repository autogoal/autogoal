import io
import os
import pathlib
import shutil
import statistics
from pathlib import Path
from typing import Dict, List, Tuple
import dill as pickle
import zipfile

import numpy as np

from autogoal.kb import Pipeline, SemanticType, build_pipeline_graph
from autogoal.ml.metrics import accuracy
from autogoal.search import PESearch
from autogoal.utils import generate_production_dockerfile, nice_repr, create_zip_file


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
        remote_sources: List[Tuple[str, int] or str] = None,
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
        self.remote_sources = remote_sources
        self.search_kwargs = search_kwargs
        self._unpickled = False
        self.export_path = None

        if random_state:
            np.random.seed(random_state)

    def _check_fitted(self):
        if not hasattr(self, "best_pipeline_"):
            raise TypeError(
                "This operation cannot be performed on an unfitted AutoML instance. Call `fit` first."
            )

    def make_pipeline_builder(self):
        if self.registry is None:
            try:
                from autogoal_contrib import find_classes, find_remote_classes
            except:
                raise ImportError(
                    "Algorithm registry nor provided and Contrib support not installed. To enable automatic code instrospection install basic contrib support running pip install autogoal-contrib"
                )

            registry = find_classes(
                include=self.include_filter, exclude=self.exclude_filter
            )

            # update registry with remote algorithms if available
            if self.remote_sources is not None:
                remote_registry = find_remote_classes(
                    sources=self.remote_sources,
                    include=self.include_filter,
                    exclude=self.exclude_filter,
                )
                registry += remote_registry

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

        self.best_pipeline_.send("train")
        self.best_pipeline_.run(X, y)
        self.best_pipeline_.send("eval")

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

        tmp = self.best_pipeline_.algorithms
        contribs = self.best_pipeline_.save_algorithms(save_path)
        self.best_pipeline_.algorithms = []
        with open(save_path / "model.bin", "wb") as fd:
            self.save(fd)
        self.best_pipeline_.algorithms = tmp

        generate_production_dockerfile(path, [contrib.split('autogoal_')[1] for contrib in contribs])

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
    def folder_load(self, path: Path = None) -> "AutoML":
        """
        Deserializes an AutoML instance from a given path.

        After deserialization, the best pipeline found is ready to predict.
        """
        load_path = path / "storage"
        with open(load_path / "model.bin", "rb") as fd:
            automl = self.load(fd)
        
        algorithms, input_types = Pipeline.load_algorithms(load_path)
        automl.best_pipeline_.algorithms = algorithms
        automl.best_pipeline_.input_types = input_types
        automl.export_path = path
        return automl

    def score(self, X, y):
        self._check_fitted()

        y_pred = self.best_pipeline_.run(X, np.zeros_like(y))
        return self.score_metric(y, y_pred)

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

    def predict(self, X):
        self._check_fitted()

        return self.best_pipeline_.run(X, None)

    def export(self, name):
        """
        Exports the result of the AutoML run onto a new Docker image.
        """
        self.save()
        os.system(
            "docker build --file ./dockerfiles/production/dockerfile-safe -t autogoal:production ."
        )
        os.system(f"docker save -o {name}.tar autogoal:production")

    def export_portable(self, path=None, generate_zip=False):
        """
        Generates a portable set of files that can be used to export the model into a new Docker image.

        :param path: Optional. The path where the generated portable set of files will be saved. If not specified, the files will be saved to the current working directory.
        :param generate_zip: Optional. A boolean value that determines whether a zip file should be generated with the exported assets. If True, a zip file will be generated and its path will be returned.
        :return: If generate_zip is False, the path to the assets directory. If generate_zip is True, the path of the generated zip file containing the exported assets.
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

serve:
	docker run -p 8000:8000 autogoal:production

        """
        )
        makefile.close()

        if generate_zip:
            filename = create_zip_file(datapath, "production_assets")
            datapath = f"{path}/{filename}.zip"

        print("generated assets for production deployment")

        return datapath
