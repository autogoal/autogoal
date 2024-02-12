import math
from typing import List


def non_dominated_sort(scores, maximize):
    """Returns scores grouped by their domination level"""
    fronts: List[List[int]] = [[]]
    domination_rank = [0] * len(scores)
    dominated_scores = [list() for _ in scores]

    for i, score_i in enumerate(scores):
        for j, score_j in enumerate(scores):
            if dominates(score_i, score_j, maximize):
                dominated_scores[i].append(j)
            elif dominates(score_j, score_i, maximize):
                domination_rank[i] += 1
        if domination_rank[i] == 0:
            fronts[0].append(i)

    front_rank = 0
    while len(fronts[front_rank]) > 0:
        next_front = []
        for i in fronts[front_rank]:
            for dominated in dominated_scores[i]:
                domination_rank[dominated] -= 1
                if domination_rank[dominated] == 0:
                    next_front.append(dominated)
        front_rank += 1
        fronts.append(next_front)

    return fronts[:-1]


def dominates(x, y, maximize) -> bool:
    """Returns true if x dominates y"""
    # print(x, y, maximize)
    assert len(x) == len(y) == len(maximize), "Mismatch between lengths"

    not_worst = all(
        x_i >= y_i if m_i else x_i <= y_i for x_i, y_i, m_i in zip(x, y, maximize)
    )
    better = any(
        (x_i > y_i if m_i else x_i < y_i) for x_i, y_i, m_i in zip(x, y, maximize)
    )
    return not_worst and better


def feature_scaling(solutions_scores: List[List[float]]) -> List[List[float]]:
    total_metrics = len(solutions_scores[0])
    scaled_scores = [list() for _ in solutions_scores]

    metric_selector = 0
    while metric_selector < total_metrics:
        # All scores per solution
        # sol1: [1, 2]
        # sol2: [3, 4]
        # m_score[0] -> [1, 3]
        # m_score[1] -> [3, 4]
        m_scores = [score[metric_selector] for score in solutions_scores]
        if len(m_scores) == 1:
            for scaled in scaled_scores:
                scaled.append(1)
            metric_selector += 1
            continue

        filtered_m_scores = [v for v in m_scores if v != -math.inf]
        if len(filtered_m_scores) == 0:
            for scaled in scaled_scores:
                scaled.append(-math.inf)
            metric_selector += 1
            continue

        max_value = max(filtered_m_scores)
        min_value = min(filtered_m_scores)
        diff = max_value - min_value

        # When there is just one valid solution (everyone else is minus infinity)
        if diff == 0:
            index = m_scores.index(max_value)
            for i, scaled in enumerate(scaled_scores):
                if i == index or m_scores[i] != -math.inf:
                    scaled.append(1)
                else:
                    scaled.append(-math.inf)
            metric_selector += 1
            continue

        for i, scaled in enumerate(scaled_scores):
            scaled_value = (
                m_scores[i] - min_value
            ) / diff  # if m_scores[i] != -math.inf else 0
            scaled.append(scaled_value)
        metric_selector += 1

    return scaled_scores
