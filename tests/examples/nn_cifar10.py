# # Solving the MEDDOCAN challenge with Keras

# This script runs an instance of [`AutoML`](/api/autogoal.ml#automl)

# In this example we only want to test the Neural Architecture Search (NAS) capabilities of AutoGOAL.

# | Dataset | URL |
# |--|--|
# | CIFAR| <https://> |

from autogoal.utils import Hour, Min
from autogoal.ml import AutoML
from autogoal.search import (
    RichLogger,
    PESearch,
)
from autogoal.kb import *

from autogoal.contrib import find_classes

# ## Experimentation

# Instantiate the classifier.
# Note that the input and output types here are defined to match the problem statement,
# i.e., entity recognition.

classifier = AutoML(
    search_algorithm=PESearch,
    input=(Tensor4, Supervised[VectorCategorical]),
    output=VectorCategorical,
    cross_validation_steps=1,
    # Since we only want to try neural networks, we restrict 
    # the contrib registry to algorithms matching with `Keras`.
    registry= find_classes("Keras"),
    errors='raise',
    # Since image classifiers are heavy to train, let's give them a longer timeout...
    evaluation_timeout=5 * Min,
    search_timeout=1 * Hour,
)

# Basic logging configuration.

loggers = [RichLogger()]

# Finally, loading the CIFAR dataset, running the `AutoML` instance,
# and printing the results.

from autogoal.datasets import cifar10

X_train, y_train, X_test, y_test = cifar10.load()


classifier.fit(X_train, y_train, logger=loggers)
score = classifier.score(X_test, y_test)

print(score)
