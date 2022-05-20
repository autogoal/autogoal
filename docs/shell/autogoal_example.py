# import high-level API
from autogoal.ml import AutoML
from autogoal.kb import MatrixContinuousDense, VectorCategorical

# load data
from autogoal.datasets import cars

X, y = cars.load()

# instantiate AutoML class
automl = AutoML(
    input=MatrixContinuousDense,
    output=VectorCategorical,
    # ... other parameters and constraints
)

# fit the model
automl.fit(X, y)

# save the best model
with open("model.bin", "wb") as fp:
    automl.save(fp)
