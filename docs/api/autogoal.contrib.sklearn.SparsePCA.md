# `autogoal.contrib.sklearn.SparsePCA`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L326)
> `SparsePCA(self, ridge_alpha, method)`

Sparse Principal Components Analysis (SparsePCA)

Finds the set of sparse components that can optimally reconstruct
the data.  The amount of sparseness is controllable by the coefficient
of the L1 penalty, given by the parameter alpha.

Read more in the :ref:`User Guide <SparsePCA>`.

Parameters
----------
n_components : int,
    Number of sparse atoms to extract.

alpha : float,
    Sparsity controlling parameter. Higher values lead to sparser
    components.

ridge_alpha : float,
    Amount of ridge shrinkage to apply in order to improve
    conditioning when calling the transform method.

max_iter : int,
    Maximum number of iterations to perform.

tol : float,
    Tolerance for the stopping condition.

method : {'lars', 'cd'}
    lars: uses the least angle regression method to solve the lasso problem
    (linear_model.lars_path)
    cd: uses the coordinate descent method to compute the
    Lasso solution (linear_model.Lasso). Lars will be faster if
    the estimated components are sparse.

n_jobs : int or None, optional (default=None)
    Number of parallel jobs to run.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

U_init : array of shape (n_samples, n_components),
    Initial values for the loadings for warm restart scenarios.

V_init : array of shape (n_components, n_features),
    Initial values for the components for warm restart scenarios.

verbose : int
    Controls the verbosity; the higher, the more messages. Defaults to 0.

random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

normalize_components : 'deprecated'
    This parameter does not have any effect. The components are always
    normalized.

    .. versionadded:: 0.20

    .. deprecated:: 0.22
       ``normalize_components`` is deprecated in 0.22 and will be removed
       in 0.24.

Attributes
----------
components_ : array, [n_components, n_features]
    Sparse components extracted from the data.

error_ : array
    Vector of errors at each iteration.

n_iter_ : int
    Number of iterations run.

mean_ : array, shape (n_features,)
    Per-feature empirical mean, estimated from the training set.
    Equal to ``X.mean(axis=0)``.

Examples
--------
>>> import numpy as np
>>> from sklearn.datasets import make_friedman1
>>> from sklearn.decomposition import SparsePCA
>>> X, _ = make_friedman1(n_samples=200, n_features=30, random_state=0)
>>> transformer = SparsePCA(n_components=5, random_state=0)
>>> transformer.fit(X)
SparsePCA(...)
>>> X_transformed = transformer.transform(X)
>>> X_transformed.shape
(200, 5)
>>> # most values in the components_ are zero (sparsity)
>>> np.mean(transformer.components_ == 0)
0.9666...

See also
--------
PCA
MiniBatchSparsePCA
DictionaryLearning
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/decomposition/_sparse_pca.py#L152)
> `fit(self, X, y=None)`

Fit the model from data in X.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training vector, where n_samples in the number of samples
    and n_features is the number of features.

y : Ignored

Returns
-------
self : object
    Returns the instance itself.
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

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L335)
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/decomposition/_sparse_pca.py#L203)
> `transform(self, X)`

Least Squares projection of the data onto the sparse components.

To avoid instability issues in case the system is under-determined,
regularization can be applied (Ridge regression) via the
`ridge_alpha` parameter.

Note that Sparse PCA components orthogonality is not enforced as in PCA
hence one cannot use a simple linear projection.

Parameters
----------
X : array of shape (n_samples, n_features)
    Test data to be transformed, must have the same number of
    features as the data used to train the model.

Returns
-------
X_new array, shape (n_samples, n_components)
    Transformed data.
