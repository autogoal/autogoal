# `autogoal.contrib.sklearn.NMF`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L256)
> `NMF(self, alpha, l1_ratio, shuffle)`

Non-Negative Matrix Factorization (NMF)

Find two non-negative matrices (W, H) whose product approximates the non-
negative matrix X. This factorization can be used for example for
dimensionality reduction, source separation or topic extraction.

The objective function is::

    0.5 * ||X - WH||_Fro^2
    + alpha * l1_ratio * ||vec(W)||_1
    + alpha * l1_ratio * ||vec(H)||_1
    + 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2
    + 0.5 * alpha * (1 - l1_ratio) * ||H||_Fro^2

Where::

    ||A||_Fro^2 = \sum_{i,j} A_{ij}^2 (Frobenius norm)
    ||vec(A)||_1 = \sum_{i,j} abs(A_{ij}) (Elementwise L1 norm)

For multiplicative-update ('mu') solver, the Frobenius norm
(0.5 * ||X - WH||_Fro^2) can be changed into another beta-divergence loss,
by changing the beta_loss parameter.

The objective function is minimized with an alternating minimization of W
and H.

Read more in the :ref:`User Guide <NMF>`.

Parameters
----------
n_components : int or None
    Number of components, if n_components is not set all features
    are kept.

init : None | 'random' | 'nndsvd' |  'nndsvda' | 'nndsvdar' | 'custom'
    Method used to initialize the procedure.
    Default: None.
    Valid options:

    - None: 'nndsvd' if n_components <= min(n_samples, n_features),
        otherwise random.

    - 'random': non-negative random matrices, scaled with:
        sqrt(X.mean() / n_components)

    - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
        initialization (better for sparseness)

    - 'nndsvda': NNDSVD with zeros filled with the average of X
        (better when sparsity is not desired)

    - 'nndsvdar': NNDSVD with zeros filled with small random values
        (generally faster, less accurate alternative to NNDSVDa
        for when sparsity is not desired)

    - 'custom': use custom matrices W and H

solver : 'cd' | 'mu'
    Numerical solver to use:
    'cd' is a Coordinate Descent solver.
    'mu' is a Multiplicative Update solver.

    .. versionadded:: 0.17
       Coordinate Descent solver.

    .. versionadded:: 0.19
       Multiplicative Update solver.

beta_loss : float or string, default 'frobenius'
    String must be in {'frobenius', 'kullback-leibler', 'itakura-saito'}.
    Beta divergence to be minimized, measuring the distance between X
    and the dot product WH. Note that values different from 'frobenius'
    (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
    fits. Note that for beta_loss <= 0 (or 'itakura-saito'), the input
    matrix X cannot contain zeros. Used only in 'mu' solver.

    .. versionadded:: 0.19

tol : float, default: 1e-4
    Tolerance of the stopping condition.

max_iter : integer, default: 200
    Maximum number of iterations before timing out.

random_state : int, RandomState instance or None, optional, default: None
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

alpha : double, default: 0.
    Constant that multiplies the regularization terms. Set it to zero to
    have no regularization.

    .. versionadded:: 0.17
       *alpha* used in the Coordinate Descent solver.

l1_ratio : double, default: 0.
    The regularization mixing parameter, with 0 <= l1_ratio <= 1.
    For l1_ratio = 0 the penalty is an elementwise L2 penalty
    (aka Frobenius Norm).
    For l1_ratio = 1 it is an elementwise L1 penalty.
    For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.

    .. versionadded:: 0.17
       Regularization parameter *l1_ratio* used in the Coordinate Descent
       solver.

verbose : bool, default=False
    Whether to be verbose.

shuffle : boolean, default: False
    If true, randomize the order of coordinates in the CD solver.

    .. versionadded:: 0.17
       *shuffle* parameter used in the Coordinate Descent solver.

Attributes
----------
components_ : array, [n_components, n_features]
    Factorization matrix, sometimes called 'dictionary'.

n_components_ : integer
    The number of components. It is same as the `n_components` parameter
    if it was given. Otherwise, it will be same as the number of
    features.

reconstruction_err_ : number
    Frobenius norm of the matrix difference, or beta-divergence, between
    the training data ``X`` and the reconstructed data ``WH`` from
    the fitted model.

n_iter_ : int
    Actual number of iterations.

Examples
--------
>>> import numpy as np
>>> X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
>>> from sklearn.decomposition import NMF
>>> model = NMF(n_components=2, init='random', random_state=0)
>>> W = model.fit_transform(X)
>>> H = model.components_

References
----------
Cichocki, Andrzej, and P. H. A. N. Anh-Huy. "Fast local algorithms for
large scale nonnegative matrix and tensor factorizations."
IEICE transactions on fundamentals of electronics, communications and
computer sciences 92.3: 708-721, 2009.

Fevotte, C., & Idier, J. (2011). Algorithms for nonnegative matrix
factorization with the beta-divergence. Neural Computation, 23(9).
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/decomposition/_nmf.py#L1298)
> `fit(self, X, y=None, **params)`

Learn a NMF model for the data X.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Data matrix to be decomposed

y : Ignored

Returns
-------
self
### `fit_transform`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/decomposition/_nmf.py#L1256)
> `fit_transform(self, X, y=None, W=None, H=None)`

Learn a NMF model for the data X and returns the transformed data.

This is more efficient than calling fit followed by transform.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Data matrix to be decomposed

y : Ignored

W : array-like, shape (n_samples, n_components)
    If init='custom', it is used as initial guess for the solution.

H : array-like, shape (n_components, n_features)
    If init='custom', it is used as initial guess for the solution.

Returns
-------
W : array, shape (n_samples, n_components)
    Transformed data.
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/decomposition/_nmf.py#L1340)
> `inverse_transform(self, W)`

Transform data back to its original space.

Parameters
----------
W : {array-like, sparse matrix}, shape (n_samples, n_components)
    Transformed data matrix

Returns
-------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Data matrix of original shape

.. versionadded:: 0.18
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L266)
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/decomposition/_nmf.py#L1315)
> `transform(self, X)`

Transform the data X according to the fitted NMF model

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Data matrix to be transformed by the model

Returns
-------
W : array, shape (n_samples, n_components)
    Transformed data
