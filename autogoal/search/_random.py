import random

from ._base import SearchAlgorithm
from autogoal.grammar import Sampler


class RandomSearch(SearchAlgorithm):
    def __init__(self, *args, random_state: int = None, **kwargs):
        super(RandomSearch, self).__init__(*args, **kwargs)
        self._sampler = Sampler(random_state=random_state)

    def _run_one_generation(self):
        yield self._generator_fn(self._sampler)
