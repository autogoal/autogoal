# `autogoal.ml`

## Submodules

* [autogoal.ml.metrics](/api/autogoal.ml.metrics/)

---

## Classes

### `AutoML`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/ml/__init__.py#L22)
> `AutoML(self, input=None, output=None, random_state=None, search_algorithm=<class 'autogoal.search._pge.PESearch'>, search_kwargs={}, search_iterations=100, include_filter='.*', exclude_filter=None, validation_split=0.3, errors='warn', cross_validation='median', cross_validation_steps=3, registry=None, score_metric=None)`

Predefined pipeline search with automatic type inference.

An `AutoML` instance represents a general-purpose machine learning
algorithm, that can be applied to any input and output.
#### `fit`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/ml/__init__.py#L64)
> `fit(self, X, y, **kwargs)`


!!! warning
    This class has no docstrings.

#### `predict`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/ml/__init__.py#L136)
> `predict(self, X)`


!!! warning
    This class has no docstrings.

#### `score`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/ml/__init__.py#L93)
> `score(self, X, y)`


!!! warning
    This class has no docstrings.


---

## Functions

### `accuracy`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/ml/metrics.py#L32)
> `accuracy(ytrue, ypred)`


!!! warning
    This class has no docstrings.


---
