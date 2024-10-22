import datetime
import inspect
import json
from typing import Any, Dict, List, Optional
from autogoal.kb._algorithm import Algorithm, AlgorithmBase, Pipeline
from autogoal.meta_learning._experience import Experience, ExperienceStore
from autogoal.meta_learning.feature_extraction._base import FeatureExtractor
from autogoal.search._base import Logger
import numpy as np

def sanitize_for_json(value):
    """
    Recursively sanitizes a value to ensure it is JSON serializable.
    
    Parameters:
        value: The value to sanitize.
    
    Returns:
        A JSON-serializable version of the value.
    """
    if value is None:
        return None
    elif isinstance(value, (str, int, float, bool)):
        return value
    elif isinstance(value, (list, tuple, set)):
        return [sanitize_for_json(item) for item in value]
    elif isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}
    elif isinstance(value, (AlgorithmBase, Pipeline)):
        # For algorithms and pipelines, extract their info recursively
        return extract_algorithm_info(value) if isinstance(value, Algorithm) else extract_algorithms_from_pipeline(value)
    else:
        # Attempt to serialize; if fails, exclude the value
        try:
            json.dumps(value)
            return value
        except TypeError:
            # Exclude the value
            return None
    
def extract_algorithms_from_pipeline(pipeline: Pipeline) -> List[Dict[str, Any]]:
    """
    Extracts algorithms and their parameters from the given pipeline in a recursive manner,
    capturing only the parameters defined in each algorithm's __init__ method.

    Parameters:
        pipeline (Pipeline): The pipeline object to extract information from.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries where each dictionary has the algorithm class name
                              as the key and its sanitized parameters as the value, recursively.
    """
    if pipeline is None:
        return []

    algorithms_info = []

    for alg in pipeline.algorithms:
        alg_info = extract_algorithm_info(alg)
        algorithms_info.append(alg_info)

    return algorithms_info

def extract_algorithm_info(alg: Algorithm) -> Dict[str, Any]:
    """
    Recursively extracts information from an algorithm, capturing only the parameters defined in its __init__ method.

    Parameters:
        alg (Algorithm): The algorithm to extract information from.

    Returns:
        Dict[str, Any]: A dictionary with the algorithm class name as key and its sanitized parameters as value.
    """
    class_name = alg.__class__.__name__
    parameters = {}

    # Retrieve the signature of the __init__ method
    signature = alg.__class__.get_inner_signature()
    for param_name, param_obj in signature.parameters.items():
        if param_name in ["self", "args", "kwargs"]:
            continue

        annotation_cls = param_obj.annotation
        if annotation_cls == inspect.Parameter.empty:
            continue
        
        value = getattr(alg, param_name, None)
        sanitized_value = sanitize_for_json(value)
        if sanitized_value is not None:
            if hasattr(annotation_cls, "__name__"):
                parameters[param_name] = {
                    "annotation": annotation_cls.__name__,
                    "value": sanitized_value,
                } 
            else:
                parameters[param_name] = {
                    "value": sanitized_value,
                }
        else:
            pass
        
    return { class_name: parameters }


class ExperienceLogger(Logger):
    def __init__(
        self,
        dataset_features=None,
        system_features=None,
        dataset_feature_extractor_name=None,
        system_feature_extractor_name=None,
        alias: str = 'default',
    ) -> None:
        self.dataset_features = dataset_features
        self.system_features = system_features
        self.dataset_feature_extractor_name = dataset_feature_extractor_name
        self.system_feature_extractor_name = system_feature_extractor_name
        self.alias = alias

    def begin(self, generations, pop_size):
        pass

    def start_generation(self, generations, best_solutions, best_fns):
        pass

    def update_best(
        self,
        solution,
        fn,
        new_best_solutions,
        best_solutions,
        new_best_fns,
        best_fns,
        new_dominated_solutions,
    ):
        pass

    def error(self, e: Exception, solution):
        # Log the experience with error information
        self.log_experience(
            solution=solution,
            f1_score=None,
            evaluation_time=None,
            accuracy=None,
            error=e,
        )

    def eval_solution(self, solution, fitness, observations):
        # Log the successful experience
        f1_score = fitness[0]  # Adjust according to your fitness structure
        evaluation_time = fitness[1]  # Adjust accordingly
        accuracy = None
        if observations is not None and "Accuracy" in observations:
            accuracy = observations["Accuracy"]

        self.log_experience(
            solution=solution,
            f1_score=f1_score,
            evaluation_time=evaluation_time,
            accuracy=accuracy,
            error=None,
        )

    def end(self, best_solutions, best_fns):
        pass

    def append_scores(self, scores, best_solutions):
        pass

    def log_experience(
        self,
        solution,
        f1_score: Optional[float],
        accuracy: Optional[float],
        evaluation_time: Optional[float],
        error: Optional[Exception] = None,
    ):
        if solution is None:
            return

        # Extract algorithms information from the solution (pipeline)
        algorithms_info = extract_algorithms_from_pipeline(solution)
        if not algorithms_info:
            return

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create an Experience instance
        experience = Experience(
            algorithms=algorithms_info,
            dataset_features=self.dataset_features,
            system_features=self.system_features,
            dataset_feature_extractor_name=self.dataset_feature_extractor_name,
            system_feature_extractor_name=self.system_feature_extractor_name,
            timestamp=timestamp,
            alias=self.alias,
            cross_val_steps=None,  # Replace if applicable
            f1=f1_score,
            evaluation_time=evaluation_time,
            accuracy=accuracy,
        )

        # Store error information if any
        if error is not None:
            experience.error = str(error)
            # Optionally set metrics to None
            experience.accuracy = None
            experience.f1 = None
            experience.evaluation_time = None

        # Save the experience
        ExperienceStore.save_experience(experience)