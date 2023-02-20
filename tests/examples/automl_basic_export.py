# AutoGOAL Example: basic usage of the AutoML class

from autogoal.datasets import dorothea
from autogoal.ml import AutoML
from autogoal.kb import *

# Load dataset
X_train, y_train, X_test, y_test = dorothea.load()

# Instantiate AutoML and define input/output types
automl = AutoML(
    input=(MatrixContinuousDense, Supervised[VectorCategorical]),
    output=VectorCategorical,
    search_iterations=1,
    remote_sources=["remote-sklearn"],
)

# Run the pipeline search process
automl.fit(X_train[0:10], y_train[0:10])

# Report the best pipeline
print(automl.best_pipeline_)
print(automl.best_score_)

# Export the result of the search process onto a brand new image called "AutoGOAL-Cars"
# automl.export_portable()
