# `autogoal.contrib.sklearn.Isomap`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1078)
> `Isomap(self, n_neighbors, n_components, eigen_solver, tol, path_method, neighbors_algorithm, p)`

Isomap Embedding

Non-linear dimensionality reduction through Isometric Mapping

Read more in the :ref:`User Guide <isomap>`.

Parameters
----------
n_neighbors : integer
    number of neighbors to consider for each point.

n_components : integer
    number of coordinates for the manifold

eigen_solver : ['auto'|'arpack'|'dense']
    'auto' : Attempt to choose the most efficient solver
    for the given problem.

    'arpack' : Use Arnoldi decomposition to find the eigenvalues
    and eigenvectors.

    'dense' : Use a direct solver (i.e. LAPACK)
    for the eigenvalue decomposition.

tol : float
    Convergence tolerance passed to arpack or lobpcg.
    not used if eigen_solver == 'dense'.

max_iter : integer
    Maximum number of iterations for the arpack solver.
    not used if eigen_solver == 'dense'.

path_method : string ['auto'|'FW'|'D']
    Method to use in finding shortest path.

    'auto' : attempt to choose the best algorithm automatically.

    'FW' : Floyd-Warshall algorithm.

    'D' : Dijkstra's algorithm.

neighbors_algorithm : string ['auto'|'brute'|'kd_tree'|'ball_tree']
    Algorithm to use for nearest neighbors search,
    passed to neighbors.NearestNeighbors instance.

n_jobs : int or None, default=None
    The number of parallel jobs to run.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

metric : string, or callable, default="minkowski"
    The metric to use when calculating distance between instances in a
    feature array. If metric is a string or callable, it must be one of
    the options allowed by :func:`sklearn.metrics.pairwise_distances` for
    its metric parameter.
    If metric is "precomputed", X is assumed to be a distance matrix and
    must be square. X may be a :term:`Glossary <sparse graph>`.

    .. versionadded:: 0.22

p : int, default=2
    Parameter for the Minkowski metric from
    sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
    equivalent to using manhattan_distance (l1), and euclidean_distance
    (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    .. versionadded:: 0.22

metric_params : dict, default=None
    Additional keyword arguments for the metric function.

    .. versionadded:: 0.22

Attributes
----------
embedding_ : array-like, shape (n_samples, n_components)
    Stores the embedding vectors.

kernel_pca_ : object
    :class:`~sklearn.decomposition.KernelPCA` object used to implement the
    embedding.

nbrs_ : sklearn.neighbors.NearestNeighbors instance
    Stores nearest neighbors instance, including BallTree or KDtree
    if applicable.

dist_matrix_ : array-like, shape (n_samples, n_samples)
    Stores the geodesic distance matrix of training data.

Examples
--------
>>> from sklearn.datasets import load_digits
>>> from sklearn.manifold import Isomap
>>> X, _ = load_digits(return_X_y=True)
>>> X.shape
(1797, 64)
>>> embedding = Isomap(n_components=2)
>>> X_transformed = embedding.fit_transform(X[:100])
>>> X_transformed.shape
(100, 2)

References
----------

.. [1] Tenenbaum, J.B.; De Silva, V.; & Langford, J.C. A global geometric
       framework for nonlinear dimensionality reduction. Science 290 (5500)
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/manifold/_isomap.py#L201)
> `fit(self, X, y=None)`

Compute the embedding vectors for data X

Parameters
----------
X : {array-like, sparse graph, BallTree, KDTree, NearestNeighbors}
    Sample data, shape = (n_samples, n_features), in the form of a
    numpy array, sparse graph, precomputed tree, or NearestNeighbors
    object.

y : Ignored

Returns
-------
self : returns an instance of self.
### `fit_transform`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/manifold/_isomap.py#L220)
> `fit_transform(self, X, y=None)`

Fit the model from data in X and transform X.

Parameters
----------
X : {array-like, sparse graph, BallTree, KDTree}
    Training vector, where n_samples in the number of samples
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
### `reconstruction_error`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/manifold/_isomap.py#L177)
> `reconstruction_error(self)`

Compute the reconstruction error for the embedding.

Returns
-------
reconstruction_error : float

Notes
-----
The cost function of an isomap embedding is

``E = frobenius_norm[K(D) - K(D_fit)] / n_samples``

Where D is the matrix of distances for the input data X,
D_fit is the matrix of distances for the output embedding X_fit,
and K is the isomap kernel:

``K(D) = -0.5 * (I - 1/n_samples) * D^2 * (I - 1/n_samples)``
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1101)
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/manifold/_isomap.py#L238)
> `transform(self, X)`

Transform X.

This is implemented by linking the points X into the graph of geodesic
distances of the training data. First the `n_neighbors` nearest
neighbors of X are found in the training data, and from these the
shortest geodesic distances from each point in X to each point in
the training data are computed in order to construct the kernel.
The embedding of X is the projection of this kernel onto the
embedding vectors of the training set.

Parameters
----------
X : array-like, shape (n_queries, n_features)
    If neighbors_algorithm='precomputed', X is assumed to be a
    distance matrix or a sparse graph of shape
    (n_queries, n_samples_fit).

Returns
-------
X_new : array-like, shape (n_queries, n_components)
