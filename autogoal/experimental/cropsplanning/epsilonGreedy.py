from autogoal.grammar import ContinuousValue, DiscreteValue, CategoricalValue
from autogoal.kb import Seq, AlgorithmBase, VectorDiscrete
from cropsPlanningV2 import AlgorithmStructure, GameStructure


class EpsilonGreedy(AlgorithmBase):

    def __init__(
            self,
            epsilon: ContinuousValue(0.0, 1),
    ):
        self._epsilon = epsilon

    def run(self, algorithm: AlgorithmStructure) -> Seq[VectorDiscrete]:
        algorithm.planning(self._epsilon)

        return algorithm.maxRoad()


