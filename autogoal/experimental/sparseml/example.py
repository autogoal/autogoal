from autogoal.ml import AutoML
from autogoal.search import RichLogger
from autogoal.kb import *
from autogoal.contrib import find_classes
from autogoal.experimental.sparseml.main import build_sparseml_keras_classifier

classifier = AutoML(
    input=(MatrixContinuousDense, Supervised[VectorCategorical]),
    output=VectorCategorical,
    cross_validation_steps=1,
    registry=[build_sparseml_keras_classifier("recipe.yaml")],
)

loggers = [RichLogger()]

from autogoal.datasets import cars
from sklearn.model_selection import train_test_split

X, y = cars.load()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classifier.fit(X_train, y_train, logger=loggers)

score = classifier.score(X_test, y_test)

print(score)
