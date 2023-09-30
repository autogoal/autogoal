# AutoGOAL Example: basic usage of the AutoML class
from autogoal.datasets import cars, dorothea
from autogoal.kb import MatrixContinuousSparse, Supervised, VectorCategorical
from autogoal.ml import AutoML, calinski_harabasz_score, silhouette_score, accuracy
from autogoal.utils import Min, Gb, Hour, Sec
from autogoal.search import PESearch, JsonLogger, ConsoleLogger
from autogoal_sklearn import AffinityPropagation, Birch, KMeans

# Load dataset
X_train, y_train, X_test, y_test = dorothea.load()

for i in range(3):
    print()
    print(f"round {i}")
    print()

    automl = AutoML(
        # Declare the input and output types
        input=(MatrixContinuousSparse, Supervised[VectorCategorical]),
        output=VectorCategorical,

        # Search space configuration
        search_timeout=1*Hour,
        evaluation_timeout= 30 * Sec,
        memory_limit=4*Gb,
        validation_split=0.3,
        cross_validation_steps=2,
    )

    # Run the pipeline search process
    automl.fit(X_train, y_train, logger=[JsonLogger(f"log-thesis-{i}.json"), ConsoleLogger()])

    # Report the best pipelines
    print(automl.best_pipelines_)
    print(automl.score(X_test, y_test))
 