# # Predefined Pipelines

# AutoGOAL is first and foremost a framework for Automatic Machine Learning.
# As such, it comes pre-packaged with hundreds of low-level machine learning
# algorithms that can be automatically assembled into pipelines for different problems.

# The core of this functionality lies in the [`AutoML`](/api/autogoal.ml#automl) class.

# To illustrate the simplicity of its use, we will first load a dataset.

from autogoal.datasets import cars
X, y = cars.load()

# Next, we import and instantiate the [`AutoML`](/api/autogoal.ml#automl) class.

from autogoal.ml import AutoML
automl = AutoML(errors='ignore')

# Finally, we just call its `fit` method. AutoGOAL will automatically infer the input and
# output type.

automl.fit(X, y)

# Sensible defaults are defined for each of the many parameters of `AutoML`.
# Make sure to read the documentation for more information on the parameters.
