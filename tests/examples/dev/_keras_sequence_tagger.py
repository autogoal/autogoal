from autogoal.contrib.keras import KerasSequenceTagger
from autogoal.datasets import cars
from autogoal.kb import CategoricalVector, Tensor3, List
from autogoal.ml import AutoML
from autogoal.search import ConsoleLogger, ProgressLogger

import numpy as np


classifier = KerasSequenceTagger(decode='crf', optimizer='sgd', epochs=20, early_stop=100).sample()


X = [
    np.asarray([[1, 1, 1, 1], [0, 1, 0, 1], [2, 1, 3, 4]]),
    np.asarray([[1, 0, 1, 1], [3, 1, 0, 1], [2, 1, 3, 4], [0, 1, 3, 4], [2, 2, 1, 4]]),
    np.asarray([[1, 1, 1, 2], [4, 1, 0, 1]]),
]

y = [
    ["A", "B", "C"],
    ["B", "A", "A", "B", "C"],
    ["C", "B"],
]


classifier.fit(X, y)
print(classifier.predict(X))
