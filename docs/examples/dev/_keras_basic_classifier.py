from autogoal.contrib.keras import KerasClassifier
from autogoal.datasets import cars
from autogoal.kb import CategoricalVector, MatrixContinuousDense
from autogoal.ml import AutoML
from autogoal.search import ConsoleLogger, ProgressLogger


classifier = AutoML(
    input=MatrixContinuousDense(),
    registry=[KerasClassifier],
    search_kwargs=dict(memory_limit=0, evaluation_timeout=0),
)


X, y = cars.load()

classifier.fit(X, y, logger=[ConsoleLogger(), ProgressLogger()])
