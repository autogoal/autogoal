# `autogoal.contrib.sklearn.MinMaxScaler`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1428)
> `MinMaxScaler(self)`

Transform features by scaling each feature to a given range.

This estimator scales and translates each feature individually such
that it is in the given range on the training set, e.g. between
zero and one.

The transformation is given by::

    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_scaled = X_std * (max - min) + min

where min, max = feature_range.

The transformation is calculated as::

    X_scaled = scale * X + min - X.min(axis=0) * scale
    where scale = (max - min) / (X.max(axis=0) - X.min(axis=0))

This transformation is often used as an alternative to zero mean,
unit variance scaling.

Read more in the :ref:`User Guide <preprocessing_scaler>`.

Parameters
----------
feature_range : tuple (min, max), default=(0, 1)
    Desired range of transformed data.

copy : bool, default=True
    Set to False to perform inplace row normalization and avoid a
    copy (if the input is already a numpy array).

Attributes
----------
min_ : ndarray of shape (n_features,)
    Per feature adjustment for minimum. Equivalent to
    ``min - X.min(axis=0) * self.scale_``

scale_ : ndarray of shape (n_features,)
    Per feature relative scaling of the data. Equivalent to
    ``(max - min) / (X.max(axis=0) - X.min(axis=0))``

    .. versionadded:: 0.17
       *scale_* attribute.

data_min_ : ndarray of shape (n_features,)
    Per feature minimum seen in the data

    .. versionadded:: 0.17
       *data_min_*

data_max_ : ndarray of shape (n_features,)
    Per feature maximum seen in the data

    .. versionadded:: 0.17
       *data_max_*

data_range_ : ndarray of shape (n_features,)
    Per feature range ``(data_max_ - data_min_)`` seen in the data

    .. versionadded:: 0.17
       *data_range_*

n_samples_seen_ : int
    The number of samples processed by the estimator.
    It will be reset on new calls to fit, but increments across
    ``partial_fit`` calls.

Examples
--------
>>> from sklearn.preprocessing import MinMaxScaler
>>> data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
>>> scaler = MinMaxScaler()
>>> print(scaler.fit(data))
MinMaxScaler()
>>> print(scaler.data_max_)
[ 1. 18.]
>>> print(scaler.transform(data))
[[0.   0.  ]
 [0.25 0.25]
 [0.5  0.5 ]
 [1.   1.  ]]
>>> print(scaler.transform([[2, 2]]))
[[1.5 0. ]]

See also
--------
minmax_scale: Equivalent function without the estimator API.

Notes
-----
NaNs are treated as missing values: disregarded in fit, and maintained in
transform.

For a comparison of the different scalers, transformers, and normalizers,
see :ref:`examples/preprocessing/plot_all_scaling.py
<sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/preprocessing/_data.py#L319)
> `fit(self, X, y=None)`

Compute the minimum and maximum to be used for later scaling.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The data used to compute the per-feature minimum and maximum
    used for later scaling along the features axis.

y : None
    Ignored.

Returns
-------
self : object
    Fitted scaler.
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/preprocessing/_data.py#L418)
> `inverse_transform(self, X)`

Undo the scaling of X according to feature_range.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Input data that will be transformed. It cannot be sparse.

Returns
-------
Xt : array-like of shape (n_samples, n_features)
    Transformed data.
### `partial_fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/preprocessing/_data.py#L341)
> `partial_fit(self, X, y=None)`

Online computation of min and max on X for later scaling.

All of X is processed as a single batch. This is intended for cases
when :meth:`fit` is not feasible due to very large number of
`n_samples` or because X is read from a continuous stream.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The data used to compute the mean and standard deviation
    used for later scaling along the features axis.

y : None
    Ignored.

Returns
-------
self : object
    Transformer instance.
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1433)
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/preprocessing/_data.py#L396)
> `transform(self, X)`

Scale features of X according to feature_range.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Input data that will be transformed.

Returns
-------
Xt : array-like of shape (n_samples, n_features)
    Transformed data.
