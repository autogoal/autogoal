from autogoal.search import SearchAlgorithm, ModelSampler
from typing import Callable


class SurrogateFunction:
    def __init__(self, func, learning_algorithm):
        self.func = func
        self.learning_algorithm = learning_algorithm


class SurrogateSearch(SearchAlgorithm):
    def __init__(
        self,
        base_search: Callable[[], SearchAlgorithm],
        estimator,
        generation_size: int = 10,
        initial_pop_size: int = 10,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.base_search = base_search
        self.generation_size = generation_size
        self.initial_pop_size = initial_pop_size

        self.training_X = []
        self.training_y = []

    def _start_generation(self):
        pass

    def _finish_generation(self, fns):
        pass

    def _build_sampler(self):
        return ModelSampler()

    def _generate(self):
        if len(self.training_X) < self.initial_pop_size:
            return super()._generate()
