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

# ## Automatic pipeline discovery

# AutoGOAL' automatic pipeline discovery relies on suitable type annotations
# for the input and output of each algorithm.
# For this functionality to work, all possible algorithms to use should
# be defined following the "algorithm" protocol.

# For example, a typical classifier has the following interface:

from autogoal.kb import DenseMatrix, CategoricalVector

class MyClassifier:
    def run(self, input: DenseMatrix()) -> CategoricalVector():
        # implementation of the algorithm
        pass

# This protocol allows AutoGOAL to automatically connect algorithms with compatible data types.
# A set of predefined data types are available in [`autogoal.kb`](/api/autogoal.kb),
# but custom data types can be defined easily.

# Here is a representation of the current data type hierarchy:

# <a href="/guide/datatypes.png", target="_blank">
#   <img src="/guide/datatypes.png" alt="Data types in AutoGOAL">
# </a>

# !!! note
#     Click the image to see a full-size version.

# ## Bundled algorithms

# AutoGOAL bundles over 100 algorithm implementations, based on multiple machine learning
# frameworks. An exhaustive list of the available algorithms is provided next. The dependencies
# that each algorithm has are also shown. Read the section on [dependencies](/dependencies/)
# for more information.

# {!docs/guide/algorithms.md!}
