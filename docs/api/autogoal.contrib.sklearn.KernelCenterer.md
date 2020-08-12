# `autogoal.contrib.sklearn.KernelCenterer`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1415)
> `KernelCenterer(self)`

Center a kernel matrix

Let K(x, z) be a kernel defined by phi(x)^T phi(z), where phi is a
function mapping x to a Hilbert space. KernelCenterer centers (i.e.,
normalize to have zero mean) the data without explicitly computing phi(x).
It is equivalent to centering phi(x) with
sklearn.preprocessing.StandardScaler(with_std=False).

Read more in the :ref:`User Guide <kernel_centering>`.

Attributes
----------
K_fit_rows_ : array, shape (n_samples,)
    Average of each column of kernel matrix

K_fit_all_ : float
    Average of kernel matrix

Examples
--------
>>> from sklearn.preprocessing import KernelCenterer
>>> from sklearn.metrics.pairwise import pairwise_kernels
>>> X = [[ 1., -2.,  2.],
...      [ -2.,  1.,  3.],
...      [ 4.,  1., -2.]]
>>> K = pairwise_kernels(X, metric='linear')
>>> K
array([[  9.,   2.,  -2.],
       [  2.,  14., -13.],
       [ -2., -13.,  21.]])
>>> transformer = KernelCenterer().fit(K)
>>> transformer
KernelCenterer()
>>> transformer.transform(K)
array([[  5.,   0.,  -5.],
       [  0.,  14., -14.],
       [ -5., -14.,  19.]])
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/preprocessing/_data.py#L2015)
> `fit(self, K, y=None)`

Fit KernelCenterer

Parameters
----------
K : numpy array of shape [n_samples, n_samples]
    Kernel matrix.

Returns
-------
self : returns an instance of self.
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
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1420)
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/preprocessing/_data.py#L2040)
> `transform(self, K, copy=True)`

Center kernel matrix.

Parameters
----------
K : numpy array of shape [n_samples1, n_samples2]
    Kernel matrix.

copy : boolean, optional, default True
    Set to False to perform inplace computation.

Returns
-------
K_new : numpy array of shape [n_samples1, n_samples2]
