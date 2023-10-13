import io
import os
import shutil
from pathlib import Path
from typing import List, Tuple
import dill as pickle

import numpy as np

from autogoal.kb import Pipeline, SemanticType, build_pipeline_graph
from autogoal.ml.metrics import (
    accuracy,
    supervised_fitness_fn_moo,
    unsupervised_fitness_fn_moo,
)
from autogoal.search import PESearch
from autogoal.utils import (
    generate_production_dockerfile,
    nice_repr,
    create_zip_file,
    ensure_directory,
)


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
        objectives=None,
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
        self.objectives = objectives or accuracy
        self.remote_sources = remote_sources
        self.search_kwargs = search_kwargs
        self._unpickled = False
        self.export_path = None

        # If objectives were not specified as iterables then create the correct objectives object
        if not type(self.objectives) is type(tuple) and not type(
            self.objectives
        ) is type(list):
            self.objectives = (self.objectives,)

        if random_state:
            np.random.seed(random_state)

    def _check_fitted(self):
        if not hasattr(self, "best_pipelines_"):
            raise TypeError(
                "This operation cannot be performed on an unfitted AutoML instance. Call `fit` first."
            )

    def make_pipeline_builder(self):
        if self.registry is not None:
            registry = self.registry
        else:
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

        self.best_pipelines_, self.best_scores_ = search.run(
            self.search_iterations, **kwargs
        )

        self.fit_pipeline(X, y)

    def fit_pipeline(self, X, y):
        self._check_fitted()

        for pipeline in self.best_pipelines_:
            pipeline.send("train")
            pipeline.run(X, y)
            pipeline.send("eval")

    def save(self, fp: io.BytesIO, pipelines: List[Pipeline] = None):
        """
        Serializes the AutoML instance.
        """
        self._check_fitted()
        pickle.Pickler(fp).dump(self)

    def serialize(self, pipelines: List[Pipeline] = None) -> List[str]:
        """
        Serializes the AutoML instance.
        """
        # If no pipelines were specified, save all the optimal ones
        if pipelines is None:
            pipelines = self.best_pipelines_

        return [pipeline.serialize() for pipeline in pipelines]

    def deserialize(self, serialization: List[str] = None):
        """
        Serializes the AutoML instance.
        """
        # If no pipelines were specified, save all the optimal ones
        if serialization is None or len(serialization) == 0:
            raise Exception("Nothing to deserialize")

        self.best_pipelines_ = [Pipeline.deserialize(p) for p in serialization]

    def folder_save(self, path: Path, pipelines: List[Pipeline] = None):
        """
        Serialize the AutoML object and save it in a directory.
        The serialized objects include the pipelines and the algorithms they use.

        Args:
        - `path (Path)`: A Path object that specifies the directory in which to save the serialized objects.
        - `pipelines (List[Pipeline], optional)`: A list of Pipeline objects to serialize. Defaults to None.

        Returns:
        None
        """
        # Make sure the AutoML object has been fitted before saving
        self._check_fitted()

        # If no path was specified, save to the current working directory
        if path is None:
            path = os.getcwd()

        # Ensure the save directory exists
        save_path = path / "storage"
        ensure_directory(save_path)

        # If no pipelines were specified, save all the optimal ones
        if pipelines is None:
            pipelines = self.best_pipelines_

        # Set to store the used algorithm contributions
        used_contribs = set()

        # Save each pipeline and its algorithms
        for i, pipeline in enumerate(pipelines):
            # Ensure the save directory exists for each solution (pipeline)
            solution_path = save_path / f"solution_{i}"
            ensure_directory(solution_path)

            # Save all algorithm contributions used by the pipeline
            used_contribs.union(pipeline.save_algorithms(solution_path))

        # serialize the AutoML instance without pipelines
        tmp = self.best_pipelines_
        self.best_pipelines_ = []
        with open(save_path / "model.bin", "wb") as fd:
            self.save(fd)
        self.best_pipelines_ = tmp

        # Generate the Dockerfile for production use, specifying the used algorithm contributions
        generate_production_dockerfile(
            path, [contrib.split("autogoal_")[1] for contrib in used_contribs]
        )

    @staticmethod
    def load(fp: io.FileIO) -> "AutoML":
        """
        Deserializes an AutoML instance.

        After deserialization, the best pipeline found is ready to predict.
        """
        automl = pickle.Unpickler(fp).load()

        if not isinstance(automl, AutoML):
            raise ValueError("The serialized file does not contain an AutoML instance.")

        return automl

    @staticmethod
    def folder_load(path: Path = None) -> "AutoML":
        """
        Deserializes an AutoML instance from a given path.

        After deserialization, the best pipeline found is ready to predict.
        """
        if path is None:
            path = Path(os.getcwd()) / "autogoal-export"

        load_path = path / "storage"
        with open(load_path / "model.bin", "rb") as fd:
            automl = AutoML.load(fd)

        pipelines = []
        solutions_folders = [
            p for p in os.listdir(load_path) if p.startswith("solution_")
        ]
        for solution_dir in solutions_folders:
            solution_path = load_path / solution_dir
            algorithms, input_types = Pipeline.load_algorithms(solution_path)
            pipelines.append(Pipeline(algorithms, input_types))

        automl.best_pipelines_ = pipelines
        automl.export_path = path
        return automl

    def score(self, X, y=None, solution_index=None):
        """
        Compute the score of the best pipelines on the given dataset.
        """
        self._check_fitted()

        scores = []
        if solution_index is None:
            for pipeline in self.best_pipelines_:
                y_pred = pipeline.run(X, np.zeros_like(y) if y else None)
                scores.append(
                    tuple([objective(y or X, y_pred) for objective in self.objectives])
                )
        else:
            pipeline = self.best_pipelines_[0]
            y_pred = pipeline.run(X, np.zeros_like(y) if y else None)
            scores.append(
                tuple([objective(y or X, y_pred) for objective in self.objectives])
            )

        return scores

    def _input_type(self, X):
        """
        Helper function to determine the input type of the dataset.
        """
        return self.input or SemanticType.infer(X)

    def _output_type(self, y):
        """
        Helper function to determine the output type of the dataset.
        """
        return self.output or SemanticType.infer(y)

    def make_fitness_fn(self, X, y=None):
        """
        Create a fitness function to evaluate pipelines.
        """
        if not y is None:
            y = np.asarray(y)

        inner_fitness_fn = (
            unsupervised_fitness_fn_moo(self.objectives)
            if y is None
            else supervised_fitness_fn_moo(self.objectives)
        )

        def fitness_fn(pipeline):
            return inner_fitness_fn(
                pipeline,
                X,
                y,
                self.validation_split,
                self.cross_validation_steps,
                self.cross_validation,
            )

        return fitness_fn

    def predict_all(self, X):
        """
        Compute predictions for all pipelines on the given dataset.
        """
        self._check_fitted()
        return [pipeline.run(X, None) for pipeline in self.best_pipelines_]

    def predict(self, X, solution_index=0):
        """
        Compute predictions for the best pipeline on the given dataset.
        """
        self._check_fitted()

        assert solution_index < len(
            self.best_pipelines_
        ), f"Cannot find solution with index {solution_index}"

        return self.best_pipelines_[solution_index].run(X, None)

    def export(self, name):
        """
        Exports the result of the AutoML run onto a new Docker image.
        """
        self.save()
        os.system(
            "docker build --file ./dockerfiles/production/dockerfile-safe -t autogoal:production ."
        )
        os.system(f"docker save -o {name}.tar autogoal:production")

    def export_portable(
        self, path=None, pipelines: List[Pipeline] = None, generate_zip=False
    ):
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

        self.folder_save(final_path, pipelines)

        makefile = open(final_path / "makefile", "w")
        makefile.write(
            """
build:
	docker build --file ./dockerfile -t autogoal:production .
	docker save -o autogoal-prod.tar autogoal:production

serve: build
	docker run -p 8000:8000 autogoal:production

        """
        )
        makefile.close()

        if generate_zip:
            filename = create_zip_file(datapath, "production_assets")
            datapath = f"{path}/{filename}.zip"

        print("generated assets for production deployment")

        return datapath
