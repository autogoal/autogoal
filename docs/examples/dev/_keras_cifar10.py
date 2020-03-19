from autogoal.ml import AutoML
from autogoal.contrib.keras._base import KerasImageClassifier
from autogoal.datasets import cifar10
from autogoal.kb import CategoricalVector, Tensor4
from autogoal.search import ConsoleLogger, ProgressLogger

from autogoal.contrib.keras._grammars import build_grammar

automl = AutoML(
    input=Tensor4(),
    output=CategoricalVector(),
    registry=[KerasImageClassifier],
    cross_validation_steps=1,
    search_kwargs=dict(
        pop_size=20,
        search_timeout = 24 * 60 * 60,
        evaluation_timeout = 0,
        memory_limit=0,
        save=False,
    ),   
    search_iterations=1000,
)


Xtrain, ytrain, Xtest, ytest = cifar10.load()

automl.fit(Xtrain, ytrain, logger=[ConsoleLogger(), ProgressLogger()])
