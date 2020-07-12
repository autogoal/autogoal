import numpy as np

from autogoal.ml import AutoML
from autogoal.contrib.keras_hierarchical import KerasImageClassifier
from autogoal.datasets import cifar10
from autogoal.kb import CategoricalVector, Tensor4
from autogoal.search import ConsoleLogger, ProgressLogger
from autogoal.utils import Gb, Hour

automl = AutoML(
    input=Tensor4(),
    output=CategoricalVector(),
    registry=[KerasImageClassifier],
    cross_validation_steps=1,
    errors="raise",
    search_kwargs=dict(
        pop_size=30,
        search_timeout= 24 * Hour,
        evaluation_timeout=1 * Hour,
        memory_limit=60 * Gb,
        save=False,
    ),
    search_iterations=1000,
    validation_split=1/6
)

Xtrain, ytrain, Xtest, ytest = cifar10.load()
X = np.vstack((Xtrain, Xtest))
y = np.hstack((ytrain, ytest))

automl.fit(X, y, logger=[ConsoleLogger(), ProgressLogger()])
