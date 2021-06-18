# AlphaZero for gaming algorithms only support one metric:
# **win_score**
# This basically takes in a custom player, and a algorithm spitted
# player and face each other in an Arena.
# win_score is calculated over win_rate.
# To provide a train_set, we could supply a set of players.
# For example, we could supply a random player, a greedy player
# and a perfect player for Tic Tac Toe and AutoGoal will try
# to optmize the win_score over each one of them.

from typing import Any, Callable
from .GameManager import God
from .GameInterface import Game


def create_win_rate_metric(game) -> Callable[[Any, Any], float]:
    def win_rate(model_player, test_player):
        wins, _, _ = God.playMatch(100, model_player, test_player, game)
        return wins / 100

    return win_rate
