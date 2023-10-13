# AutoGOAL Example: basic usage of the AutoML class

from autogoal.datasets import dorothea
from autogoal.ml import AutoML
from autogoal.utils import Min, Gb, Hour, Sec
from autogoal.search import ConsoleLogger, RichLogger
from autogoal.kb import *

from autogoal.search._base import ConsoleLogger

# Load dataset
X_train, y_train, X_test, y_test = dorothea.load()

# Instantiate AutoML and define input/output types
automl = AutoML(
    input=(MatrixContinuousSparse, Supervised[VectorCategorical]),
    output=VectorCategorical,
)

# Run the pipeline search process
automl.fit(X_train, y_train, logger=RichLogger())

# Report the best pipelines
print(automl.best_pipelines_)
print(automl.best_scores_)

# Export the result of the search process onto a brand new image called "AutoGOAL-Cars"
automl.export_portable(generate_zip=True)

