# AutoGOAL Example: basic usage of the AutoML class
from autogoal.datasets import cars
from autogoal.kb import MatrixContinuousDense, Supervised, VectorCategorical
from autogoal.search import RichLogger
from autogoal.ml import AutoML, calinski_harabasz_score

# Load dataset
X, y = cars.load()

# Instantiate AutoML, define input/output types and the score metric
automl = AutoML(
    input=(MatrixContinuousDense, Supervised[VectorCategorical]),
    output=VectorCategorical,
    score_metric=calinski_harabasz_score,
)

# Run the pipeline search process
automl.fit(X[0:10])

# Report the best pipeline
print(automl.best_pipeline_)
print(automl.best_score_)
