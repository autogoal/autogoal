from autogoal.contrib.keras import KerasSequenceTagger
from autogoal.datasets import cars
from autogoal.kb import CategoricalVector, Tensor3, List
from autogoal.ml import AutoClassifier
from autogoal.search import ConsoleLogger, ProgressLogger

import numpy as np


classifier = KerasSequenceTagger(epochs=100, early_stop=100).sample()


X = np.asarray(
    [
        [[1, 1, 1, 1], [0, 1, 0, 1], [2, 1, 3, 4]],
        [[1, 0, 1, 1], [3, 1, 0, 1], [2, 1, 3, 4]],
        [[1, 1, 1, 2], [4, 1, 0, 1], [2, 1, 3, 4]],
    ]
)

y = np.asarray([
    ["A", "B", "C"],
    ["B", "A", "A"],
    ["C", "B", "C"],
])


classifier.fit(X, y)
print(classifier.predict(X))
