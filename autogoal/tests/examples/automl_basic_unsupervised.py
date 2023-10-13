# AutoGOAL Example: basic usage of the AutoML class
from autogoal.datasets import cars, dorothea
from autogoal.kb import MatrixContinuousSparse, VectorDiscrete
from autogoal.ml import AutoML, calinski_harabasz_score
from autogoal.utils import Min, Gb, Hour, Sec
from autogoal.search import PESearch, JsonLogger, ConsoleLogger

# Load dataset
X_train, y_train, X_test, y_test = dorothea.load()

#TODO: Fix this example. Unsupervised is not working as intended right now.

automl = AutoML(
        # Declare the input and output types
        input=MatrixContinuousSparse,
        output=VectorDiscrete,
        objectives=calinski_harabasz_score,

        # Search space configuration
        search_timeout=120*Sec,
        evaluation_timeout= 10 * Sec,
        memory_limit=4*Gb,
        validation_split=0.3,
        cross_validation_steps=2,
    )

# Run the pipeline search process
automl.fit(X_train, logger=ConsoleLogger())

# Report the best pipelines
print(automl.best_pipelines_)
print(automl.score(X_test, y_test))