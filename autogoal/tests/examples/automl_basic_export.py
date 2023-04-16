# AutoGOAL Example: basic usage of the AutoML class

from autogoal.datasets import dorothea
from autogoal.ml import AutoML
from autogoal.kb import *

from autogoal.search._base import ConsoleLogger

# Load dataset
X_train, y_train, X_test, y_test = dorothea.load()

# import autogoal_remote.production.client as client

# response = client.post_eval(X_test)

# total = len(response)
# ok_count = 0
# for i in range(len(response)):
#     ok_count = ok_count + 1 if y_test[i] == response[i] else ok_count
# print(f"ok: {ok_count} out of {total}")

# Instantiate AutoML and define input/output types
automl = AutoML(
    input=(MatrixContinuousSparse, Supervised[VectorCategorical]),
    output=VectorCategorical,
    search_iterations=5,
    search_timeout=60,
    evaluation_timeout=10
    # remote_sources=["remote-sklearn"],
)

# Run the pipeline search process
automl.fit(X_train, y_train, logger=ConsoleLogger())

# Report the best pipeline
print(automl.best_pipeline_)
print(automl.best_score_)

# Export the result of the search process onto a brand new image called "AutoGOAL-Cars"
# automl.export_portable()

automl.export_portable(generate_zip=True)
