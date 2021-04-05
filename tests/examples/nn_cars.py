# # Solving the CARS dataset with Keras

# This script runs an instance of [`AutoML`](/api/autogoal.ml#automl) on the CARS dataset.
# In this example we only want to test the Neural Architecture Search (NAS) capabilities of AutoGOAL.
# CARS is a classic numeric feature dataset.

# | Dataset | URL |
# |--|--|
# | Cars | <https://archive.ics.uci.edu/ml/datasets/Car+Evaluation> |

# As usual, we will need the `AutoML` class, a suitable logger, and semantic types from `autogoal.kb`.

from autogoal.ml import AutoML
from autogoal.search import RichLogger
from autogoal.kb import *

# To restrict which types of algorithms can `AutoML` use, we will manually invoke `find_classes`.

from autogoal.contrib import find_classes

# ## Experimentation

# Instantiate the classifier.
# Note that the input and output types here are defined to match the problem statement,
# i.e., supervised classification from matrix-like features.

classifier = AutoML(
    input=(MatrixContinuousDense, Supervised[VectorCategorical]),
    output=VectorCategorical,
    # We will set `cross_validation_steps=1` to reduce the time that we spend on each pipeline.
    # Keep in mind this will increase the generalization error of the AutoML process.
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

# By default, this will run for 5 minutes.

classifier.fit(X_train, y_train, logger=loggers)
score = classifier.score(X_test, y_test)

# Let's see what we got!

print(score)
