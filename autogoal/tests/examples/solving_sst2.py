# AutoGOAL Example: basic usage of the AutoML class

from autogoal.datasets import sst2
from autogoal.kb import Seq, Sentence, Supervised, VectorCategorical
from autogoal.ml import AutoML
from autogoal.search._base import ConsoleLogger

# Load dataset
X, y = sst2.load()

# Instantiate AutoML and define input/output types
automl = AutoML(
    input=(Seq[Sentence], Supervised[VectorCategorical]),
    output=VectorCategorical,
)

# Run the pipeline search process
automl.fit(X[:100], y[:100], logger=ConsoleLogger())

# Report the best pipeline
print(automl.best_pipelines_)
print(automl.best_scores_)
