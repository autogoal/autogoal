# `autogoal.contrib.sklearn.FastICA`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L181)
> `FastICA(self, algorithm, whiten, fun)`

FastICA: a fast algorithm for Independent Component Analysis.

Read more in the :ref:`User Guide <ICA>`.

Parameters
----------
n_components : int, optional
    Number of components to use. If none is passed, all are used.

algorithm : {'parallel', 'deflation'}
    Apply parallel or deflational algorithm for FastICA.

whiten : boolean, optional
    If whiten is false, the data is already considered to be
    whitened, and no whitening is performed.

fun : string or function, optional. Default: 'logcosh'
    The functional form of the G function used in the
    approximation to neg-entropy. Could be either 'logcosh', 'exp',
    or 'cube'.
    You can also provide your own function. It should return a tuple
    containing the value of the function, and of its derivative, in the
    point. Example:

    def my_g(x):
        return x ** 3, (3 * x ** 2).mean(axis=-1)

fun_args : dictionary, optional
    Arguments to send to the functional form.
    If empty and if fun='logcosh', fun_args will take value
    {'alpha' : 1.0}.

max_iter : int, optional
    Maximum number of iterations during fit.

tol : float, optional
    Tolerance on update at each iteration.

w_init : None of an (n_components, n_components) ndarray
    The mixing matrix to be used to initialize the algorithm.

random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

Attributes
----------
components_ : 2D array, shape (n_components, n_features)
    The linear operator to apply to the data to get the independent
    sources. This is equal to the unmixing matrix when ``whiten`` is
    False, and equal to ``np.dot(unmixing_matrix, self.whitening_)`` when
    ``whiten`` is True.

mixing_ : array, shape (n_features, n_components)
    The pseudo-inverse of ``components_``. It is the linear operator
    that maps independent sources to the data.

mean_ : array, shape(n_features)
    The mean over features. Only set if `self.whiten` is True.

n_iter_ : int
    If the algorithm is "deflation", n_iter is the
    maximum number of iterations run across all components. Else
    they are just the number of iterations taken to converge.

whitening_ : array, shape (n_components, n_features)
    Only set if whiten is 'True'. This is the pre-whitening matrix
    that projects data onto the first `n_components` principal components.

Examples
--------
>>> from sklearn.datasets import load_digits
>>> from sklearn.decomposition import FastICA
>>> X, _ = load_digits(return_X_y=True)
>>> transformer = FastICA(n_components=7,
...         random_state=0)
>>> X_transformed = transformer.fit_transform(X)
>>> X_transformed.shape
(1797, 7)

Notes
-----
Implementation based on
*A. Hyvarinen and E. Oja, Independent Component Analysis:
Algorithms and Applications, Neural Networks, 13(4-5), 2000,
pp. 411-430*
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/decomposition/_fastica.py#L561)
> `fit(self, X, y=None)`

Fit the model to X.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training data, where n_samples is the number of samples
    and n_features is the number of features.

y : Ignored

Returns
-------
self
### `fit_transform`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/decomposition/_fastica.py#L544)
> `fit_transform(self, X, y=None)`

Fit the model and recover the sources from X.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training data, where n_samples is the number of samples
    and n_features is the number of features.

y : Ignored

Returns
-------
X_new : array-like, shape (n_samples, n_components)
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/decomposition/_fastica.py#L603)
> `inverse_transform(self, X, copy=True)`

Transform the sources back to the mixed data (apply mixing matrix).

Parameters
----------
X : array-like, shape (n_samples, n_components)
    Sources, where n_samples is the number of samples
    and n_components is the number of components.
copy : bool (optional)
    If False, data passed to fit are overwritten. Defaults to True.

Returns
-------
X_new : array-like, shape (n_samples, n_features)
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L191)
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/decomposition/_fastica.py#L579)
> `transform(self, X, copy=True)`

Recover the sources from X (apply the unmixing matrix).

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Data to transform, where n_samples is the number of samples
    and n_features is the number of features.

copy : bool (optional)
    If False, data passed to fit are overwritten. Defaults to True.

Returns
-------
X_new : array-like, shape (n_samples, n_components)
