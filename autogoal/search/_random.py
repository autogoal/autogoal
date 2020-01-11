import random

from ._base import SearchAlgorithm
from autogoal.grammar import Sampler


class RandomSearch(SearchAlgorithm):
    def __init__(self, generator_fn, fitness_fn, *, random_state: int = None, **kwargs):
        super(RandomSearch, self).__init__(generator_fn, fitness_fn, **kwargs)
        self._sampler = Sampler(random_state=random_state)

    def _run_one_generation(self):
        yield self._generator_fn(self._sampler)
