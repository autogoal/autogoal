# `autogoal.contrib.sklearn.RobustScaler`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1454)
> `RobustScaler(self, with_centering, with_scaling)`

Scale features using statistics that are robust to outliers.

This Scaler removes the median and scales the data according to
the quantile range (defaults to IQR: Interquartile Range).
The IQR is the range between the 1st quartile (25th quantile)
and the 3rd quartile (75th quantile).

Centering and scaling happen independently on each feature by
computing the relevant statistics on the samples in the training
set. Median and interquartile range are then stored to be used on
later data using the ``transform`` method.

Standardization of a dataset is a common requirement for many
machine learning estimators. Typically this is done by removing the mean
and scaling to unit variance. However, outliers can often influence the
sample mean / variance in a negative way. In such cases, the median and
the interquartile range often give better results.

.. versionadded:: 0.17

Read more in the :ref:`User Guide <preprocessing_scaler>`.

Parameters
----------
with_centering : boolean, True by default
    If True, center the data before scaling.
    This will cause ``transform`` to raise an exception when attempted on
    sparse matrices, because centering them entails building a dense
    matrix which in common use cases is likely to be too large to fit in
    memory.

with_scaling : boolean, True by default
    If True, scale the data to interquartile range.

quantile_range : tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0
    Default: (25.0, 75.0) = (1st quantile, 3rd quantile) = IQR
    Quantile range used to calculate ``scale_``.

    .. versionadded:: 0.18

copy : boolean, optional, default is True
    If False, try to avoid a copy and do inplace scaling instead.
    This is not guaranteed to always work inplace; e.g. if the data is
    not a NumPy array or scipy.sparse CSR matrix, a copy may still be
    returned.

Attributes
----------
center_ : array of floats
    The median value for each feature in the training set.

scale_ : array of floats
    The (scaled) interquartile range for each feature in the training set.

    .. versionadded:: 0.17
       *scale_* attribute.

Examples
--------
>>> from sklearn.preprocessing import RobustScaler
>>> X = [[ 1., -2.,  2.],
...      [ -2.,  1.,  3.],
...      [ 4.,  1., -2.]]
>>> transformer = RobustScaler().fit(X)
>>> transformer
RobustScaler()
>>> transformer.transform(X)
array([[ 0. , -2. ,  0. ],
       [-1. ,  0. ,  0.4],
       [ 1. ,  0. , -1.6]])

See also
--------
robust_scale: Equivalent function without the estimator API.

:class:`sklearn.decomposition.PCA`
    Further removes the linear correlation across features with
    'whiten=True'.

Notes
-----
For a comparison of the different scalers, transformers, and normalizers,
see :ref:`examples/preprocessing/plot_all_scaling.py
<sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.

https://en.wikipedia.org/wiki/Median
https://en.wikipedia.org/wiki/Interquartile_range
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/preprocessing/_data.py#L1188)
> `fit(self, X, y=None)`

Compute the median and quantiles to be used for scaling.

Parameters
----------
X : array-like, shape [n_samples, n_features]
    The data used to compute the median and quantiles
    used for later scaling along the features axis.
### `fit_transform`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/base.py#L544)
> `fit_transform(self, X, y=None, **fit_params)`

Fit to data, then transform it.

Fits transformer to X and y with optional parameters fit_params
and returns a transformed version of X.

Parameters
----------
X : numpy array of shape [n_samples, n_features]
    Training set.

y : numpy array of shape [n_samples]
    Target values.

**fit_params : dict
    Additional fit parameters.

Returns
-------
X_new : numpy array of shape [n_samples, n_features_new]
    Transformed array.
### `get_params`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/base.py#L173)
> `get_params(self, deep=True)`

Get parameters for this estimator.

Parameters
----------
deep : bool, default=True
    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

Returns
-------
params : mapping of string to any
    Parameter names mapped to their values.
### `inverse_transform`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/preprocessing/_data.py#L1262)
> `inverse_transform(self, X)`

Scale back the data to the original representation

Parameters
----------
X : array-like
    The data used to scale along the specified axis.
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1461)
> `run(self, input)`

### `set_params`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/base.py#L205)
> `set_params(self, **params)`

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as pipelines). The latter have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

Parameters
----------
**params : dict
    Estimator parameters.

Returns
-------
self : object
    Estimator instance.
### `train`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L47)
> `train(self)`

### `transform`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/preprocessing/_data.py#L1239)
> `transform(self, X)`

Center and scale the data.

Parameters
----------
X : {array-like, sparse matrix}
    The data used to scale along the specified axis.
