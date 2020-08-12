# `autogoal.contrib.sklearn.MiniBatchSparsePCA`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L299)
> `MiniBatchSparsePCA(self, ridge_alpha, n_iter, batch_size, shuffle, method)`

Mini-batch Sparse Principal Components Analysis

Finds the set of sparse components that can optimally reconstruct
the data.  The amount of sparseness is controllable by the coefficient
of the L1 penalty, given by the parameter alpha.

Read more in the :ref:`User Guide <SparsePCA>`.

Parameters
----------
n_components : int,
    number of sparse atoms to extract

alpha : int,
    Sparsity controlling parameter. Higher values lead to sparser
    components.

ridge_alpha : float,
    Amount of ridge shrinkage to apply in order to improve
    conditioning when calling the transform method.

n_iter : int,
    number of iterations to perform for each mini batch

callback : callable or None, optional (default: None)
    callable that gets invoked every five iterations

batch_size : int,
    the number of features to take in each mini batch

verbose : int
    Controls the verbosity; the higher, the more messages. Defaults to 0.

shuffle : boolean,
    whether to shuffle the data before splitting it in batches

n_jobs : int or None, optional (default=None)
    Number of parallel jobs to run.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

method : {'lars', 'cd'}
    lars: uses the least angle regression method to solve the lasso problem
    (linear_model.lars_path)
    cd: uses the coordinate descent method to compute the
    Lasso solution (linear_model.Lasso). Lars will be faster if
    the estimated components are sparse.

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

n_iter_ : int
    Number of iterations run.

mean_ : array, shape (n_features,)
    Per-feature empirical mean, estimated from the training set.
    Equal to ``X.mean(axis=0)``.

Examples
--------
>>> import numpy as np
>>> from sklearn.datasets import make_friedman1
>>> from sklearn.decomposition import MiniBatchSparsePCA
>>> X, _ = make_friedman1(n_samples=200, n_features=30, random_state=0)
>>> transformer = MiniBatchSparsePCA(n_components=5, batch_size=50,
...                                  random_state=0)
>>> transformer.fit(X)
MiniBatchSparsePCA(...)
>>> X_transformed = transformer.transform(X)
>>> X_transformed.shape
(200, 5)
>>> # most values in the components_ are zero (sparsity)
>>> np.mean(transformer.components_ == 0)
0.94

See also
--------
PCA
SparsePCA
DictionaryLearning
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/decomposition/_sparse_pca.py#L350)
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

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L318)
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
