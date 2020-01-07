import random

from ._base import SearchAlgorithm
from autogoal.grammar import Sampler


class RandomSearch(SearchAlgorithm):
    def __init__(self, grammar, fitness_fn, *, random_state: int = None):
        super(RandomSearch, self).__init__(grammar, fitness_fn)
        self._sampler = Sampler(random_state=random_state)

    def _run_one_generation(self):
        yield self._grammar.sample(sampler=self._sampler)
