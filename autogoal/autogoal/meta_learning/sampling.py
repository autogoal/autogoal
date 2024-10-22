from enum import Enum
from typing import Dict, List, Any, Tuple
from autogoal.meta_learning._experience import Experience
from autogoal.sampling import (
    DistributionParam,
    MeanDevParam,
    ModelParam,
    ModelSampler,
    UnormalizedWeightParam,
    WeightParam,
)
import numpy as np

from typing import Dict, List, Any, Tuple
from autogoal.meta_learning._experience import Experience
from autogoal.sampling import (
    DistributionParam,
    MeanDevParam,
    ModelParam,
    ModelSampler,
    UnormalizedWeightParam,
    WeightParam,
)
import numpy as np


class ExperienceSamplerMode(Enum):
    Replicate = "replicate"
    RegularSample = "regular-sample"


class ExperienceReplayModelSampler(ModelSampler):
    """
    A ModelSampler that adjusts its internal probabilistic model based on external meta-knowledge,
    effectively 'warming up' towards configurations that are more likely to perform well.

    This class inherits from ModelSampler and adds methods to adjust the model based on past experiences.
    """

    def __init__(
        self,
        model: Dict = None,
        mode: ExperienceSamplerMode = ExperienceSamplerMode.RegularSample,
        current_experience: "Experience" = None,
        **kwargs,
    ):
        self.current_experience = current_experience
        self.mode = mode
        super().__init__(model=model, **kwargs)

    def set_replicate_mode(self, experience: "Experience"):
        self.current_experience = experience
        self.mode = ExperienceSamplerMode.Replicate

    def set_regular_sample_mode(self):
        self.current_experience = None
        self.mode = ExperienceSamplerMode.RegularSample
        
    def _resolve_handle(self, handle, algorithm_experience: Dict[str, Dict]):
        splitted_handle = handle.split("_")
        if len(splitted_handle) > 1: # handle is a parameter of an algorithm that might be nested
            algorithm = splitted_handle[0]
            parameter_name = "_".join(splitted_handle[1:])
            for algorithm_name, params in algorithm_experience.items():
                if algorithm_name == algorithm: # We found the algorithm, now we look for the parameter
                    next_iteration = []
                    for param in params.keys():
                        if parameter_name.startswith(param): # Possible match
                            if param == parameter_name: # We found the parameter
                                if 'annotation' in params[param]:
                                    return list(params[param]['value'].keys())[0]
                                else:
                                    return params[param]['value']
                                
                            else: # No full match might mean that the parameter is nested
                                if 'annotation' in params[param]:
                                    next_iteration.append(params[param]['value'])
                                
                    # If we didn't find the parameter, we go to the nested iteration
                    for next_iter in next_iteration:
                        result = self._resolve_handle(parameter_name, next_iter)
                        if result is not None:
                            return result
        else:
            for algorithm_name, params in algorithm_experience.items():
                if handle == algorithm_name:
                    return algorithm_name
                
                for param in params.keys():
                    if 'annotation' in params[param]:
                        if handle == params[param]["annotation"]:
                            return list(params[param]['value'].keys())[0]
                    
        return None

    def _get_value_from_experience(self, handle, experience: Experience, options: List=None):
        if handle is None: # If handle is None, we are looking for an algorithm in the experience
            for algorithm in experience.algorithms: # priority to initial algorithms
                algorithm_name = list(algorithm.keys())[0] # Should be only one key
                if algorithm_name in options:
                    return algorithm_name
        
        else: # If handle is not None, we are looking for a specific parameter of some algorithm
            for algorithm in experience.algorithms:
                resolved_handle = self._resolve_handle(handle, algorithm)
                
                if resolved_handle is not None:
                    if options is not None:
                        if hasattr(options[0], "name"): # If options are Symbol objects
                            option_names = [option.name for option in options]
                            if (resolved_handle in option_names):
                                return options[option_names.index(resolved_handle)]
                        else:
                            # Options are not Symbol objects
                            if resolved_handle in options:
                                return resolved_handle
                    else:
                        return resolved_handle
            
            # if not handle is None: # If handle is not None, we are looking for a specific parameter
            #     parameter_value = None
            #     for parameter in experience.finetuning_parameters:
            #         if handle.endswith(parameter):
            #             parameter_value = experience.finetuning_parameters[parameter]
            #             break
                
            #     if parameter_value is not None:
            #         return experience.finetuning_parameters[parameter]
            #     else:
            #         if hasattr(options[0], "name"): # If options are Symbol objects
            #             option_names = [option.name for option in options]
            #             if experience.finetuning_parameters["inner_model"] in option_names:
            #                 return options[option_names.index(experience.finetuning_parameters["inner_model"])]
                
            # else: # If handle is None, we are looking for a specific finetuning method
            #     if experience.finetuning_method in options:
            #         return experience.finetuning_method
            
            return None

    def choice(self, options, handle=None):
        if handle is not None:
            return self.categorical(options, handle)
        
        weights = [
            self._get_model_params(option, UnormalizedWeightParam(value=1))
            for option in options
        ]
        
        if self.mode == ExperienceSamplerMode.Replicate and self.current_experience is not None:
            expected_value = self._get_value_from_experience(handle, self.current_experience, options)
            if expected_value is not None:
                self._register_update(expected_value, 1)
                return expected_value
        
        idx = self.rand.choices(
            range(len(options)), weights=[w.value for w in weights], k=1
        )[0]
        
        option = options[idx]
        self._register_update(option, 1)
        return option
    
    def discrete(self, min=0, max=10, handle=None):
        params = self._get_model_params(
            handle, MeanDevParam(mean=(min + max) / 2, dev=(max - min))
        )
        
        if self.mode == ExperienceSamplerMode.Replicate and self.current_experience is not None:
            expected_value = self._get_value_from_experience(handle, self.current_experience)
            if expected_value is not None and min <= expected_value <= max:
                self._register_update(handle, expected_value)
                return expected_value
        
        params = self._get_model_params(
            handle, MeanDevParam(mean=(min + max) / 2, dev=(max - min))
        )
        
        value = self._clamp(int(self.rand.gauss(params.mean, params.dev)), min, max)
        return self._register_update(handle, value)

    def continuous(self, min=0, max=1, handle=None):
        if self.mode == ExperienceSamplerMode.Replicate and self.current_experience is not None:
            expected_value = self._get_value_from_experience(handle, self.current_experience)
            if expected_value is not None and min <= expected_value <= max:
                self._register_update(handle, expected_value)
                return expected_value

        params = self._get_model_params(
            handle, MeanDevParam(mean=(min + max) / 2, dev=(max - min))
        )
        value = self._clamp(self.rand.gauss(params.mean, params.dev), min, max)
        return self._register_update(handle, value)

    def boolean(self, handle=None):
        params = self._get_model_params(handle, WeightParam(value=0.5))
        
        if self.mode == ExperienceSamplerMode.Replicate and self.current_experience is not None:
            expected_value = self._get_value_from_experience(handle, self.current_experience)
            if expected_value is not None and isinstance(expected_value, bool):
                self._register_update(handle, expected_value)
                return expected_value

        value = self.rand.uniform(0, 1) < params.value
        return self._register_update(handle, value)

    def categorical(self, options, handle=None):
        params = self._get_model_params(
            handle, DistributionParam(weights=[1 for _ in options])
        )
        
        if self.mode == ExperienceSamplerMode.Replicate and self.current_experience is not None:
            expected_value = self._get_value_from_experience(handle, self.current_experience, options)
            if expected_value is not None:
                idx = options.index(expected_value)
                self._register_update(handle, idx)
                return expected_value

        idx = self.rand.choices(range(len(options)), weights=params.weights, k=1)[0]
        return options[self._register_update(handle, idx)]
