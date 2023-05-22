# AutoGOAL Example: basic usage of the AutoML class
from autogoal.datasets import cars
from autogoal.kb import MatrixContinuousDense, Supervised, VectorDiscrete
from autogoal.search import RichLogger, ConsoleLogger
from autogoal.ml import AutoML, calinski_harabasz_score, silhouette_score
from autogoal.utils import Min

# Load dataset
X, y = cars.load()

# Instantiate AutoML, define input/output types and the score metric
automl = AutoML(
    input=MatrixContinuousDense,
    output=VectorDiscrete,
    objectives=silhouette_score,
    evaluation_timeout= 1/2 * Min,
)

# Run the pipeline search process
automl.fit(X, logger=ConsoleLogger())

# Report the best pipeline
print(automl.best_pipelines_)
print(automl.score(X))
 