from autogoal.ml import AutoML
from autogoal.contrib.keras._base import KerasClassifier
from autogoal.datasets import cifar10
from autogoal.kb import CategoricalVector
from autogoal.search import ConsoleLogger, ProgressLogger


automl = AutoML(output=CategoricalVector(), registry=[KerasClassifier])

Xtrain, ytrain, Xtest, ytest = cifar10.load(1)

automl.fit(Xtrain, ytrain, logger=[ConsoleLogger(), ProgressLogger()])
