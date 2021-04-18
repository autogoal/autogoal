# # Solving the MEDDOCAN challenge with Keras

# This script runs an instance of [`AutoML`](/api/autogoal.ml#automl)
# in the [MEDDOCAN 2019 challenge](https://github.com/PlanTL-SANIDAD/SPACCC_MEDDOCAN).
# The full source code can be found [here](https://github.com/autogoal/autogoal/blob/main/docs/examples/solving_meddocan_2019.py).

# In this example we only want to test the Neural Architecture Search (NAS) capabilities of AutoGOAL.

# | Dataset | URL |
# |--|--|
# | MEDDOCAN 2019 | <https://github.com/PlanTL-SANIDAD/SPACCC_MEDDOCAN> |

from autogoal.ml import AutoML
from autogoal.datasets import meddocan
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
    input=(Seq[Seq[Word]], Supervised[Seq[Seq[Label]]]),
    output=Seq[Seq[Label]],
    score_metric=meddocan.F1_beta,
    cross_validation_steps=1,
    # Since we only want to try neural networks, we restrict
    # the contrib registry to algorithms matching with `Keras`.
    registry=find_classes("Keras|Bert"),
    # We need to give some extra time because neural networks are slow
    evaluation_timeout=300,
    search_timeout=1800,
)

# Basic logging configuration.

loggers = [RichLogger()]

# Finally, loading the MEDDOCAN dataset, running the `AutoML` instance,
# and printing the results.

X_train, y_train, X_test, y_test = meddocan.load()

classifier.fit(X_train, y_train, logger=loggers)
score = classifier.score(X_test, y_test)

print(score)
