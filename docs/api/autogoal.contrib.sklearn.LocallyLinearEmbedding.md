# `autogoal.contrib.sklearn.LocallyLinearEmbedding`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1111)
> `LocallyLinearEmbedding(self, n_neighbors, n_components, reg, eigen_solver, method, neighbors_algorithm)`

Locally Linear Embedding

Read more in the :ref:`User Guide <locally_linear_embedding>`.

Parameters
----------
n_neighbors : integer
    number of neighbors to consider for each point.

n_components : integer
    number of coordinates for the manifold

reg : float
    regularization constant, multiplies the trace of the local covariance
    matrix of the distances.

eigen_solver : string, {'auto', 'arpack', 'dense'}
    auto : algorithm will attempt to choose the best method for input data

    arpack : use arnoldi iteration in shift-invert mode.
                For this method, M may be a dense matrix, sparse matrix,
                or general linear operator.
                Warning: ARPACK can be unstable for some problems.  It is
                best to try several random seeds in order to check results.

    dense  : use standard dense matrix operations for the eigenvalue
                decomposition.  For this method, M must be an array
                or matrix type.  This method should be avoided for
                large problems.

tol : float, optional
    Tolerance for 'arpack' method
    Not used if eigen_solver=='dense'.

max_iter : integer
    maximum number of iterations for the arpack solver.
    Not used if eigen_solver=='dense'.

method : string ('standard', 'hessian', 'modified' or 'ltsa')
    standard : use the standard locally linear embedding algorithm.  see
               reference [1]
    hessian  : use the Hessian eigenmap method. This method requires
               ``n_neighbors > n_components * (1 + (n_components + 1) / 2``
               see reference [2]
    modified : use the modified locally linear embedding algorithm.
               see reference [3]
    ltsa     : use local tangent space alignment algorithm
               see reference [4]

hessian_tol : float, optional
    Tolerance for Hessian eigenmapping method.
    Only used if ``method == 'hessian'``

modified_tol : float, optional
    Tolerance for modified LLE method.
    Only used if ``method == 'modified'``

neighbors_algorithm : string ['auto'|'brute'|'kd_tree'|'ball_tree']
    algorithm to use for nearest neighbors search,
    passed to neighbors.NearestNeighbors instance

random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`. Used when ``eigen_solver`` == 'arpack'.

n_jobs : int or None, optional (default=None)
    The number of parallel jobs to run.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

Attributes
----------
embedding_ : array-like, shape [n_samples, n_components]
    Stores the embedding vectors

reconstruction_error_ : float
    Reconstruction error associated with `embedding_`

nbrs_ : NearestNeighbors object
    Stores nearest neighbors instance, including BallTree or KDtree
    if applicable.

Examples
--------
>>> from sklearn.datasets import load_digits
>>> from sklearn.manifold import LocallyLinearEmbedding
>>> X, _ = load_digits(return_X_y=True)
>>> X.shape
(1797, 64)
>>> embedding = LocallyLinearEmbedding(n_components=2)
>>> X_transformed = embedding.fit_transform(X[:100])
>>> X_transformed.shape
(100, 2)

References
----------

.. [1] Roweis, S. & Saul, L. Nonlinear dimensionality reduction
    by locally linear embedding.  Science 290:2323 (2000).
.. [2] Donoho, D. & Grimes, C. Hessian eigenmaps: Locally
    linear embedding techniques for high-dimensional data.
    Proc Natl Acad Sci U S A.  100:5591 (2003).
.. [3] Zhang, Z. & Wang, J. MLLE: Modified Locally Linear
    Embedding Using Multiple Weights.
    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.70.382
.. [4] Zhang, Z. & Zha, H. Principal manifolds and nonlinear
    dimensionality reduction via tangent space alignment.
    Journal of Shanghai Univ.  8:406 (2004)
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/manifold/_locally_linear.py#L669)
> `fit(self, X, y=None)`

Compute the embedding vectors for data X

Parameters
----------
X : array-like of shape [n_samples, n_features]
    training set.

y : Ignored

Returns
-------
self : returns an instance of self.
### `fit_transform`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/manifold/_locally_linear.py#L686)
> `fit_transform(self, X, y=None)`

Compute the embedding vectors for data X and transform X.

Parameters
----------
X : array-like of shape [n_samples, n_features]
    training set.

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
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1132)
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/manifold/_locally_linear.py#L703)
> `transform(self, X)`

Transform new points into embedding space.

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
X_new : array, shape = [n_samples, n_components]

Notes
-----
Because of scaling performed by this method, it is discouraged to use
it together with methods that are not scale-invariant (like SVMs)
