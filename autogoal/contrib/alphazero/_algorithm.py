from autogoal.kb import AlgorithmBase
from autogoal.grammar import ContinuousValue, CategoricalValue, DiscreteValue
from autogoal.utils import nice_repr
from ._semantics import GameStructure, Player
from .NeuralNet import NeuralNetwork
from .Agent import AlphaZeroAgent


@nice_repr
class AlphaZeroAlgorithm(AlgorithmBase):
    def __init__(
        self,
        reg_const: ContinuousValue(0.0001, 0.001),
        learning_rate: ContinuousValue(0.0001, 0.05),
        batch_size: CategoricalValue(8, 16, 32, 64, 128, 256),
        epochs: DiscreteValue(10, 100),
        num_iters: DiscreteValue(10, 100),
        queue_len: DiscreteValue(20000, 40000),
        episodes: DiscreteValue(50, 100),
        memory_size: DiscreteValue(3000, 10000),
        arena_games: DiscreteValue(40, 100),
        update_threshold: ContinuousValue(0.6, 0.99),
        time_limit: DiscreteValue(5 * 60, 60 * 60 * 24) # Define a time limit of 5 minutes - 1 day
    ) -> None:
        """
        AlphaZero algorithm representation for arbitrary games
        implementing the Game Interface.

        WARN: *time_limit* param should be configured on demand,
        inheriting from AlphaZeroAlgorithm and overriding init call
        to accept all parameters except time_limit if is not desire
        that AutoGoal try all different values for optimization.
        """
        self.reg_const = reg_const
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_iters = num_iters
        self.queue_len = queue_len
        self.episodes = episodes
        self.memory_size = memory_size
        self.arena_games = arena_games
        self.update_threshold = update_threshold
        self.time_limit = time_limit
        self._mode = "train"
        self.agent = None

    def train(self):
        self._mode = "train"

    def eval(self):
        self._mode = "eval"

    def fit(self, X):
        cnn = NeuralNetwork(
            self.reg_const, self.learning_rate, self.batch_size, self.epochs, X
        )

        self.agent = AlphaZeroAgent(X, cnn)

        self.agent.train(
            num_iters=self.num_iters,
            arena_games=self.arena_games,
            episodes=self.episodes,
            queue_len=self.queue_len,
            update_threshold=self.update_threshold,
            time_limit=self.time_limit
        )

    def run(self, game: GameStructure) -> Player:
        if self._mode == "train":
           self.fit(game)

        if self._mode == "eval":
            if self.agent is None:
                raise TypeError(
                "You must call `fit` before `predict` to learn class mappings."
            )
        
            return self.agent