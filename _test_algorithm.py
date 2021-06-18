from autogoal.ml import AutoML
from autogoal.contrib.alphazero._metrics import create_win_rate_metric
from autogoal.contrib.alphazero.TicTacToeGame import TicTacToeGame
from autogoal.contrib.alphazero.ramdomAgent import RandomPlayer
from autogoal.contrib.alphazero._semantics import GameStructure, Player
from autogoal.contrib.alphazero._algorithm import AlphaZeroAlgorithm

game = TicTacToeGame()

test_player = RandomPlayer(game)

win_rate = create_win_rate_metric(game)

automl = AutoML(
    input=GameStructure,
    output=Player,
    score_metric=win_rate,
    registry=[AlphaZeroAlgorithm]
)

automl.fit(X=game, y=test_player)

score = automl.score(game, test_player)

print(score)