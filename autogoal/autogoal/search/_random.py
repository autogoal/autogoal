import random

from ._base import SearchAlgorithm
from autogoal.sampling import Sampler


class RandomSearch(SearchAlgorithm):
    def __init__(self, *args, random_state: int = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._sampler = Sampler(random_state=random_state)

    def _build_sampler(self):
        return self._sampler
