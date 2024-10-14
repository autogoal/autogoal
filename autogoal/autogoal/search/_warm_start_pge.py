from typing import Optional, Dict, List, Type
from autogoal.meta_learning.sampling import ExperienceReplayModelSampler
from autogoal.sampling import ModelSampler, ReplaySampler, best_indices, merge_updates, update_model
from autogoal.search._base import SearchAlgorithm
from autogoal.meta_learning import ExperienceStore, Experience
from autogoal.meta_learning import WarmStart
from autogoal.search._nspge import NSPESearch

class NSPEWarmStartSearch(NSPESearch):
    """
    A variant of NSPESearch that integrates warm start functionality
    using past experiences to adjust the initial probabilistic model.
    """

    def __init__(
        self,
        *args,
        warm_start: Optional[WarmStart] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.warm_start = warm_start if warm_start is not None else WarmStart(None, 0.2)
        self.warm_started = False
        
        
    def _start_generation(self):
        if (
            not self.warm_started
            and self.warm_start
        ):
            self.warm_start.warm_up(generator_fn=self._generator_fn)
            self._model = self.warm_start._model
            self.warm_started = True
            
        return super()._start_generation()
    

    def _build_sampler(self):
        if len(self._samplers) < self._epsilon_greed * self._pop_size:
            if self.warm_started:
                sampler = ModelSampler(
                    self._model, random_state=self._random_states.getrandbits(32)
                )
            else:
                sampler = ModelSampler(random_state=self._random_states.getrandbits(32))
        else:
            sampler = ModelSampler(
                    self._model, random_state=self._random_states.getrandbits(32)
                )

        self._samplers.append(sampler)
        return sampler