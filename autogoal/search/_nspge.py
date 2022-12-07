import math
from typing import List, Optional
from autogoal.search.moo_utils import feature_scaling, non_dominated_sort
from autogoal.utils import Gb, Min, Sec
from ._pge import PESearch


# TODO: scale function results in crowding distance
# TODO: Return multiple possible pipelines, instead of just the one
# TODO: How is the genotype updated, since we are not using the best, but a list of the bests.
#       Does it gets updated *N* times with the *N* fittest??
# TODO: Where/When is the population cropped
class NSPESearch(PESearch):
    def __init__(
        self,
        generator_fn=None,
        fitness_fn=None,
        pop_size=20,
        maximize=True,
        errors="raise",
        early_stop=0.5,
        evaluation_timeout: int = 10 * Sec,
        memory_limit: int = 4 * Gb,
        search_timeout: int = 5 * Min,
        target_fn=None,
        allow_duplicates=True,
        number_of_solutions=None,
        ranking_fn=None,
        learning_factor=0.05,
        selection: float = 0.2,
        epsilon_greed: float = 0.1,
        random_state: Optional[int] = None,
        name: str = None,
        save: bool = False,
        **kwargs,
    ):
        def default_ranking_fn(_, fns):
            rankings = [-math.inf] * len(fns)
            fronts = non_dominated_sort(fns, self._maximize)
            # return fronts[0]
            for ranking, front in enumerate(fronts):
                for index in front:
                    rankings[index] = -ranking
            return rankings

        if ranking_fn is None:
            ranking_fn = default_ranking_fn

        super().__init__(
            generator_fn=generator_fn,
            fitness_fn=fitness_fn,
            pop_size=pop_size,
            maximize=maximize,
            errors=errors,
            early_stop=early_stop,
            evaluation_timeout=evaluation_timeout,
            memory_limit=memory_limit,
            search_timeout=search_timeout,
            target_fn=target_fn,
            allow_duplicates=allow_duplicates,
            number_of_solutions=number_of_solutions,
            ranking_fn=ranking_fn,
            learning_factor=learning_factor,
            selection=selection,
            epsilon_greed=epsilon_greed,
            random_state=random_state,
            name=name,
            save=save,
            **kwargs,
        )

    def _indices_of_fittest(self, fns: List[List[float]]):
        fronts = non_dominated_sort(fns, self._maximize)
        indices = []
        k = int(self._selection * len(fns))

        for front in fronts:
            if len(indices) + len(front) <= k:
                indices.extend(front)
            else:
                indices.extend(
                    sorted(front, key=lambda i: -self.crowding_distance(fns, front, i))[
                        : k - len(indices)
                    ]
                )
                break
        return indices

    def crowding_distance(
        self, scores: List[List[float]], front: List[int], index: int
    ) -> float:
        scaled_scores = feature_scaling(scores)

        crowding_distances: List[float] = [0 for _ in scores]
        for m in range(len(self._maximize)):
            front = sorted(front, key=lambda x: scores[x][m])
            crowding_distances[front[0]] = math.inf
            crowding_distances[front[-1]] = math.inf
            m_values = [scaled_scores[i][m] for i in front]
            scale: float = max(m_values) - min(m_values)
            if scale == 0:
                scale = 1
            for i in range(1, len(front) - 1):
                crowding_distances[i] += (
                    scaled_scores[front[i + 1]][m] - scaled_scores[front[i - 1]][m]
                ) / scale

        return crowding_distances[index]
