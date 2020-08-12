# `autogoal.contrib.sklearn.PowerTransformer`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1441)
> `PowerTransformer(self, standardize)`

Apply a power transform featurewise to make data more Gaussian-like.

Power transforms are a family of parametric, monotonic transformations
that are applied to make data more Gaussian-like. This is useful for
modeling issues related to heteroscedasticity (non-constant variance),
or other situations where normality is desired.

Currently, PowerTransformer supports the Box-Cox transform and the
Yeo-Johnson transform. The optimal parameter for stabilizing variance and
minimizing skewness is estimated through maximum likelihood.

Box-Cox requires input data to be strictly positive, while Yeo-Johnson
supports both positive or negative data.

By default, zero-mean, unit-variance normalization is applied to the
transformed data.

Read more in the :ref:`User Guide <preprocessing_transformer>`.

.. versionadded:: 0.20

Parameters
----------
method : str, (default='yeo-johnson')
    The power transform method. Available methods are:

    - 'yeo-johnson' [1]_, works with positive and negative values
    - 'box-cox' [2]_, only works with strictly positive values

standardize : boolean, default=True
    Set to True to apply zero-mean, unit-variance normalization to the
    transformed output.

copy : boolean, optional, default=True
    Set to False to perform inplace computation during transformation.

Attributes
----------
lambdas_ : array of float, shape (n_features,)
    The parameters of the power transformation for the selected features.

Examples
--------
>>> import numpy as np
>>> from sklearn.preprocessing import PowerTransformer
>>> pt = PowerTransformer()
>>> data = [[1, 2], [3, 2], [4, 5]]
>>> print(pt.fit(data))
PowerTransformer()
>>> print(pt.lambdas_)
[ 1.386... -3.100...]
>>> print(pt.transform(data))
[[-1.316... -0.707...]
 [ 0.209... -0.707...]
 [ 1.106...  1.414...]]

See also
--------
power_transform : Equivalent function without the estimator API.

QuantileTransformer : Maps data to a standard normal distribution with
    the parameter `output_distribution='normal'`.

Notes
-----
NaNs are treated as missing values: disregarded in ``fit``, and maintained
in ``transform``.

For a comparison of the different scalers, transformers, and normalizers,
see :ref:`examples/preprocessing/plot_all_scaling.py
<sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.

References
----------

.. [1] I.K. Yeo and R.A. Johnson, "A new family of power transformations to
       improve normality or symmetry." Biometrika, 87(4), pp.954-959,
       (2000).

.. [2] G.E.P. Box and D.R. Cox, "An Analysis of Transformations", Journal
       of the Royal Statistical Society B, 26, 211-252 (1964).
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/preprocessing/_data.py#L2776)
> `fit(self, X, y=None)`

Estimate the optimal parameter lambda for each feature.

The optimal lambda parameter for minimizing skewness is estimated on
each feature independently using maximum likelihood.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    The data used to estimate the optimal transformation parameters.

y : Ignored

Returns
-------
self : object
### `fit_transform`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/preprocessing/_data.py#L2796)
> `fit_transform(self, X, y=None)`

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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/preprocessing/_data.py#L2856)
> `inverse_transform(self, X)`

Apply the inverse power transformation using the fitted lambdas.

The inverse of the Box-Cox transformation is given by::

    if lambda_ == 0:
        X = exp(X_trans)
    else:
        X = (X_trans * lambda_ + 1) ** (1 / lambda_)

The inverse of the Yeo-Johnson transformation is given by::

    if X >= 0 and lambda_ == 0:
        X = exp(X_trans) - 1
    elif X >= 0 and lambda_ != 0:
        X = (X_trans * lambda_ + 1) ** (1 / lambda_) - 1
    elif X < 0 and lambda_ != 2:
        X = 1 - (-(2 - lambda_) * X_trans + 1) ** (1 / (2 - lambda_))
    elif X < 0 and lambda_ == 2:
        X = 1 - exp(-X_trans)

Parameters
----------
X : array-like, shape (n_samples, n_features)
    The transformed data.

Returns
-------
X : array-like, shape (n_samples, n_features)
    The original data
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1446)
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/preprocessing/_data.py#L2828)
> `transform(self, X)`

Apply the power transform to each feature using the fitted lambdas.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    The data to be transformed using a power transformation.

Returns
-------
X_trans : array-like, shape (n_samples, n_features)
    The transformed data.
