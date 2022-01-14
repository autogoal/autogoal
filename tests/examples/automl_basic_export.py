# AutoGOAL Example: basic usage of the AutoML class

from autogoal.datasets import cars
from autogoal.kb import MatrixContinuousDense, Supervised, VectorCategorical
from autogoal.ml import AutoML

# Load dataset
X, y = cars.load()

# Instantiate AutoML and define input/output types
automl = AutoML(
    input=(MatrixContinuousDense, Supervised[VectorCategorical]),
    output=VectorCategorical,
    search_iterations=1
)

# Run the pipeline search process
automl.fit(X[0:10], y[0:10])

# Report the best pipeline
print(automl.best_pipeline_)
print(automl.best_score_)

# Export the result of the search process onto a brand new image called "AutoGOAL-Cars"
automl.export_portable()