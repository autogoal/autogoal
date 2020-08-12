# `autogoal.contrib.sklearn.TruncatedSVD`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L343)
> `TruncatedSVD(self, n_components, n_iter, tol)`

Dimensionality reduction using truncated SVD (aka LSA).

This transformer performs linear dimensionality reduction by means of
truncated singular value decomposition (SVD). Contrary to PCA, this
estimator does not center the data before computing the singular value
decomposition. This means it can work with scipy.sparse matrices
efficiently.

In particular, truncated SVD works on term count/tf-idf matrices as
returned by the vectorizers in sklearn.feature_extraction.text. In that
context, it is known as latent semantic analysis (LSA).

This estimator supports two algorithms: a fast randomized SVD solver, and
a "naive" algorithm that uses ARPACK as an eigensolver on (X * X.T) or
(X.T * X), whichever is more efficient.

Read more in the :ref:`User Guide <LSA>`.

Parameters
----------
n_components : int, default = 2
    Desired dimensionality of output data.
    Must be strictly less than the number of features.
    The default value is useful for visualisation. For LSA, a value of
    100 is recommended.

algorithm : string, default = "randomized"
    SVD solver to use. Either "arpack" for the ARPACK wrapper in SciPy
    (scipy.sparse.linalg.svds), or "randomized" for the randomized
    algorithm due to Halko (2009).

n_iter : int, optional (default 5)
    Number of iterations for randomized SVD solver. Not used by ARPACK. The
    default is larger than the default in
    `~sklearn.utils.extmath.randomized_svd` to handle sparse matrices that
    may have large slowly decaying spectrum.

random_state : int, RandomState instance or None, optional, default = None
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

tol : float, optional
    Tolerance for ARPACK. 0 means machine precision. Ignored by randomized
    SVD solver.

Attributes
----------
components_ : array, shape (n_components, n_features)

explained_variance_ : array, shape (n_components,)
    The variance of the training samples transformed by a projection to
    each component.

explained_variance_ratio_ : array, shape (n_components,)
    Percentage of variance explained by each of the selected components.

singular_values_ : array, shape (n_components,)
    The singular values corresponding to each of the selected components.
    The singular values are equal to the 2-norms of the ``n_components``
    variables in the lower-dimensional space.

Examples
--------
>>> from sklearn.decomposition import TruncatedSVD
>>> from scipy.sparse import random as sparse_random
>>> from sklearn.random_projection import sparse_random_matrix
>>> X = sparse_random(100, 100, density=0.01, format='csr',
...                   random_state=42)
>>> svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
>>> svd.fit(X)
TruncatedSVD(n_components=5, n_iter=7, random_state=42)
>>> print(svd.explained_variance_ratio_)
[0.0646... 0.0633... 0.0639... 0.0535... 0.0406...]
>>> print(svd.explained_variance_ratio_.sum())
0.286...
>>> print(svd.singular_values_)
[1.553... 1.512...  1.510... 1.370... 1.199...]

See also
--------
PCA

References
----------
Finding structure with randomness: Stochastic algorithms for constructing
approximate matrix decompositions
Halko, et al., 2009 (arXiv:909) https://arxiv.org/pdf/0909.4061.pdf

Notes
-----
SVD suffers from a problem called "sign indeterminacy", which means the
sign of the ``components_`` and the output from transform depend on the
algorithm and random state. To work around this, fit instances of this
class to data once, then keep the instance around to do transformations.
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/decomposition/_truncated_svd.py#L128)
> `fit(self, X, y=None)`

Fit LSI model on training data X.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Training data.

y : Ignored

Returns
-------
self : object
    Returns the transformer object.
### `fit_transform`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/decomposition/_truncated_svd.py#L146)
> `fit_transform(self, X, y=None)`

Fit LSI model to X and perform dimensionality reduction on X.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Training data.

y : Ignored

Returns
-------
X_new : array, shape (n_samples, n_components)
    Reduced version of X. This will always be a dense array.
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/decomposition/_truncated_svd.py#L215)
> `inverse_transform(self, X)`

Transform X back to its original space.

Returns an array X_original whose transform would be X.

Parameters
----------
X : array-like, shape (n_samples, n_components)
    New data.

Returns
-------
X_original : array, shape (n_samples, n_features)
    Note that this is always a dense array.
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L353)
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/decomposition/_truncated_svd.py#L199)
> `transform(self, X)`

Perform dimensionality reduction on X.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    New data.

Returns
-------
X_new : array, shape (n_samples, n_components)
    Reduced version of X. This will always be a dense array.
