import statistics
import abc

from typing import Mapping, Optional, Dict, List, Sequence
from autogoal.sampling import ModelSampler, best_indices, merge_updates, update_model
from ._base import SearchAlgorithm

import random


class PESearch(SearchAlgorithm):
    def __init__(
        self,
        *args,
        learning_factor: float = 0.05,
        selection: float = 0.2,
        epsilon_greed: float = 0.1,
        random_state: Optional[int] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._learning_factor = learning_factor
        self._selection = selection
        self._epsilon_greed = epsilon_greed
        self._model: Dict = {}
        self._random_states = random.Random(random_state)

    def _start_generation(self):
        self._samplers = []

    def _build_sampler(self):
        if len(self._samplers) < self._epsilon_greed * self._pop_size:
            sampler = ModelSampler(random_state=self._random_states.getrandbits(32))
        else:
            sampler = ModelSampler(self._model, random_state=self._random_states.getrandbits(32))

        self._samplers.append(sampler)
        return sampler

    def _finish_generation(self, fns):
        # Compute the marginal model of the best pipelines
        indices = best_indices(
            fns, k=int(self._selection * len(fns)), maximize=self._maximize
        )
        samplers: List[ModelSampler] = [self._samplers[i] for i in indices]
        updates: Dict = merge_updates(*[sampler.updates for sampler in samplers])

        # Update the probabilistic model with the marginal model from the best pipelines
        self._model = update_model(self._model, updates, self._learning_factor)
