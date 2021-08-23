from autogoal.contrib import find_classes
from autogoal.grammar import DiscreteValue
from autogoal.kb import Seq, VectorDiscrete
from autogoal.ml import AutoML
from epsilonGreedy import EpsilonGreedy
from cropsPlanningV2 import cropsFilter
from epsilonGreedy import AlgorithmStructure


if __name__ == "__main__":

    epsilon = 1 - (0.1 + 0.1 / 4)
    number_of_neurons = 10000
    crops = 3
    planning_time = 10
    value_of_decisions = [1, 2]
    time_of_decicions = [10, 10]
    crop_rotation = [0, 0]
    neighboring_crops = [1, 0]

    algorithmStructure = cropsFilter(crops, value_of_decisions, time_of_decicions,
                crop_rotation, planning_time, number_of_neurons)

    automl = AutoML(
        input=AlgorithmStructure,
        output=Seq[VectorDiscrete],
        registry=[EpsilonGreedy] + find_classes())

    automl.fit(algorithmStructure, [1, 1, 1])

    score = automl.score(algorithmStructure, [1, 1, 1])

    print(score)



    # score = automl.
    # print(score)
