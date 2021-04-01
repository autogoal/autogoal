import numpy as np

from autogoal.ml import AutoML
from autogoal.contrib.keras import KerasImageClassifier, KerasImagePreprocessor
from autogoal.datasets import cifar10
from autogoal.kb import CategoricalVector, Tensor4
from autogoal.search import ConsoleLogger, ProgressLogger

automl = AutoML(
    input=Tensor4(),
    output=CategoricalVector(),
    registry=[KerasImageClassifier],
    # registry=[KerasImageClassifier, KerasImagePreprocessor],
    cross_validation_steps=1,
    search_kwargs=dict(
        pop_size=20,
        search_timeout=24 * 60 * 60,
        evaluation_timeout=0,
        memory_limit=0,
        save=False,
    ),
    search_iterations=1000,
    validation_split=1/6
)

Xtrain, ytrain, Xtest, ytest = cifar10.load()
X = np.vstack((Xtrain, Xtest))
y = np.hstack((ytrain, ytest))

automl.fit(X, y, logger=[ConsoleLogger(), ProgressLogger()])
