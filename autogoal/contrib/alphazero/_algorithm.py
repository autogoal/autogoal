from autogoal.kb import AlgorithmBase
from autogoal.grammar import ContinuousValue, CategoricalValue, DiscreteValue
from ._semantics import GameStructure, Player
from .NeuralNet import NeuralNetwork
from .Agent import AlphaZeroAgent


class AlphaZeroAlgorithm(AlgorithmBase):
    def __init__(
        self,
        reg_const: ContinuousValue(0.0001, 0.001),
        learning_rate: ContinuousValue(0.0001, 0.5),
        batch_size: CategoricalValue(8, 16, 32, 64, 128, 256),
        epochs: DiscreteValue(10, 100)
    ) -> None:
        self.reg_const = reg_const
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

    def run(self, game: GameStructure) -> Player:
        # First step, construct a CNN model
        # and train it with the hyperparameters
        # defined for this algorithm instance
        cnn = NeuralNetwork(self.reg_const, self.learning_rate, self.batch_size, self.epochs, game)
        agent = AlphaZeroAgent(game, cnn)

        # Train the agent
        agent.train()

        return agent