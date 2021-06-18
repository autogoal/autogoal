from autogoal.ml import AutoML
from autogoal.contrib.alphazero._metrics import create_win_rate_metric
from autogoal.contrib.alphazero.TicTacToeGame import TicTacToeGame
from autogoal.contrib.alphazero.ramdomAgent import RandomPlayer
from autogoal.contrib.alphazero._semantics import GameStructure, Player
from autogoal.contrib.alphazero._algorithm import AlphaZeroAlgorithm
from autogoal.grammar import ContinuousValue, DiscreteValue, CategoricalValue

game = TicTacToeGame()

test_player = RandomPlayer(game)

win_rate = create_win_rate_metric(game, display=game.display)

# Standard RL training is a long process, trying to improve with
# the maximun simulations posible. For a simple game like Tic Tac Toe
# we might use a very simple training process, and reduce training time to just
# 5 minutes Top


class AlphaZeroTicTacToe(AlphaZeroAlgorithm):
    def __init__(
        self,
        reg_const: ContinuousValue(0.0001, 0.001),
        learning_rate: ContinuousValue(0.0001, 0.02),
        batch_size: CategoricalValue(8, 16, 32, 64, 128, 256),
        epochs: DiscreteValue(10, 100),
        num_iters: DiscreteValue(10, 15),
        queue_len: DiscreteValue(20000, 40000),
        episodes: DiscreteValue(2, 5),
        memory_size: DiscreteValue(3000, 10000),
        arena_games: DiscreteValue(40, 100),
        update_threshold: ContinuousValue(0.5, 0.9),
        time_limit: DiscreteValue(5 * 60, 10 * 60), # limit to 5-10 minutes
    ) -> None:
        super().__init__(
            reg_const,
            learning_rate,
            batch_size,
            epochs,
            num_iters,
            queue_len,
            episodes,
            memory_size,
            arena_games,
            update_threshold,
            time_limit,
        )


automl = AutoML(
    input=GameStructure,
    output=Player,
    score_metric=win_rate,
    registry=[AlphaZeroTicTacToe],
)

automl.fit(X=game, y=test_player)

score = automl.score(game, test_player)

print(score)
