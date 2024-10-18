import datetime
from typing import Any, Dict, Optional
from autogoal.kb._algorithm import Pipeline
from autogoal.meta_learning._experience import Experience, ExperienceStore
from autogoal.meta_learning.feature_extraction._base import FeatureExtractor
from autogoal.search._base import Logger
import numpy as np

def extract_experience_from_pipeline(pipeline: Pipeline) -> Dict[str, Any]:
    """
    Extracts experience information from a given pipeline by finding the first algorithm
    whose class name includes 'FineTune' or 'Lora'.

    Parameters:
        pipeline (Pipeline): The pipeline object to extract information from.

    Returns:
        Dict[str, Any]: A dictionary containing the model class name, fine-tuning method,
                        and fine-tuning parameters.

    Raises:
        ValueError: If no fine-tuning algorithm is found in the pipeline.
    """
    # Search for the fine-tuning algorithm in the pipeline
    if pipeline is None:
        return
    
    finetune_algorithm = None
    for alg in pipeline.algorithms:
        class_name = alg.__class__.__name__.lower()
        if 'finetune' in class_name or 'lora' in class_name:
            finetune_algorithm = alg
            break

    if finetune_algorithm is None:
        return None

    # Extract the model class name
    model_class_name = finetune_algorithm.__class__.__name__

    # Extract the fine-tuning method name from the inner model, if it exists
    finetuning_method = finetune_algorithm.__class__.__name__

    # Extract fine-tuning parameters
    finetuning_parameters = {}
    for k, v in finetune_algorithm.__dict__.items():
        if not k.startswith('_'):
            if k == 'inner_model':
                finetuning_parameters['inner_model'] = v.__class__.__name__
            else:
                finetuning_parameters[k] = v

    return {
        'model_class_name': model_class_name,
        'finetuning_method': finetuning_method,
        'finetuning_parameters': finetuning_parameters,
    }

class ExperienceLogger(Logger):
    def __init__(
        self,
        dataset_features=None,
        system_features=None,
        dataset_feature_extractor_name=None,
        system_feature_extractor_name=None,
    ) -> None:
        self.dataset_features = dataset_features
        self.system_features = system_features
        self.dataset_feature_extractor_name = dataset_feature_extractor_name
        self.system_feature_extractor_name = system_feature_extractor_name

    def begin(self, generations, pop_size):
        pass  # You can log this information if needed

    def start_generation(self, generations, best_solutions, best_fns):
        pass  # Implement if you need to log at the start of each generation

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
        pass  # Implement if needed

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
        # assuming the fitness is a tuple with the f1 score and evaluation time
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
        pass  # Implement if you need to log at the end of the run

    def append_scores(self, scores, best_solutions):
        pass  # Implement if needed

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
        
        # Extract information from the solution (pipeline)
        experience_info = extract_experience_from_pipeline(solution)
        if experience_info is None:
            return
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create an Experience instance
        experience = Experience(
            model_class_name=experience_info['model_class_name'],
            finetuning_method=experience_info['finetuning_method'],
            finetuning_parameters=experience_info['finetuning_parameters'],
            dataset_features=self.dataset_features,
            system_features=self.system_features,
            dataset_feature_extractor_name=self.dataset_feature_extractor_name,
            system_feature_extractor_name=self.system_feature_extractor_name,
            timestamp=timestamp,
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
