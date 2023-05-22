import math
from typing import List
from autogoal.search.utils import feature_scaling, non_dominated_sort
from ._pge import PESearch


class NSPESearch(PESearch):
    def __init__(
        self,
        *args,
        ranking_fn=None,
        **kwargs,
    ):
        def default_ranking_fn(_, fns):
            rankings = [-math.inf] * len(fns)
            fronts = non_dominated_sort(fns, self._maximize)
            for ranking, front in enumerate(fronts):
                for index in front:
                    rankings[index] = -ranking
            return rankings

        if ranking_fn is None:
            ranking_fn = default_ranking_fn

        super().__init__(
            *args,
            ranking_fn=ranking_fn,
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