# # Solving the MEDDOCAN challenge with Keras

# This script runs an instance of [`AutoML`](/api/autogoal.ml#automl)

# In this example we only want to test the Neural Architecture Search (NAS) capabilities of AutoGOAL.

# | Dataset | URL |
# |--|--|
# | Cars | <https://archive.ics.uci.edu/ml/datasets/Car+Evaluation> |

from autogoal.ml import AutoML
from autogoal.search import RichLogger
from autogoal.kb import *

from autogoal.contrib import find_classes

# ## Experimentation

# Instantiate the classifier.
# Note that the input and output types here are defined to match the problem statement,
# i.e., entity recognition.

classifier = AutoML(
    input=(MatrixContinuousDense, Supervised[VectorCategorical]),
    output=VectorCategorical,
    cross_validation_steps=1,
    # Since we only want to try neural networks, we restrict
    # the contrib registry to algorithms matching with `Keras`.
    registry=find_classes("Keras"),
)

# Basic logging configuration.

loggers = [RichLogger()]

# Finally, loading the CARS dataset, running the `AutoML` instance,
# and printing the results.

from autogoal.datasets import cars

from sklearn.model_selection import train_test_split

X, y = cars.load()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


classifier.fit(X_train, y_train, logger=loggers)
score = classifier.score(X_test, y_test)

print(score)
