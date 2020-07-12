from autogoal.ml import AutoML
from autogoal.contrib.keras_hierarchical import *
from autogoal.kb import *
from autogoal.search import ConsoleLogger, ProgressLogger
from autogoal.utils import Gb, Hour

from autogoal.datasets import mnist

automl = AutoML(
    input=Tensor4(),
    output=CategoricalVector(),
    registry=[KerasImageClassifier],
    cross_validation_steps=1,
    search_kwargs=dict(
        pop_size=30,
        search_timeout=24 * Hour,
        evaluation_timeout=1 * Hour,
        memory_limit=60 * Gb,
        save=False,
    ),
    search_iterations=1000,
    validation_split=1/6
)

Xtrain, ytrain, xt, yt = mnist.load(unrolled=False)

automl.fit(Xtrain, ytrain, logger=[ConsoleLogger(), ProgressLogger()])