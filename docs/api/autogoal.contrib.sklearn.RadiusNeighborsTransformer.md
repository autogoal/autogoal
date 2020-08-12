# `autogoal.contrib.sklearn.RadiusNeighborsTransformer`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1282)
> `RadiusNeighborsTransformer(self, mode, radius, algorithm, leaf_size, p)`

Transform X into a (weighted) graph of neighbors nearer than a radius

The transformed data is a sparse graph as returned by
radius_neighbors_graph.

Read more in the :ref:`User Guide <neighbors_transformer>`.

.. versionadded:: 0.22

Parameters
----------
mode : {'distance', 'connectivity'}, default='distance'
    Type of returned matrix: 'connectivity' will return the connectivity
    matrix with ones and zeros, and 'distance' will return the distances
    between neighbors according to the given metric.

radius : float, default=1.
    Radius of neighborhood in the transformed sparse graph.

algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
    Algorithm used to compute the nearest neighbors:

    - 'ball_tree' will use :class:`BallTree`
    - 'kd_tree' will use :class:`KDTree`
    - 'brute' will use a brute-force search.
    - 'auto' will attempt to decide the most appropriate algorithm
      based on the values passed to :meth:`fit` method.

    Note: fitting on sparse input will override the setting of
    this parameter, using brute force.

leaf_size : int, default=30
    Leaf size passed to BallTree or KDTree.  This can affect the
    speed of the construction and query, as well as the memory
    required to store the tree.  The optimal value depends on the
    nature of the problem.

metric : string or callable, default='minkowski'
    metric to use for distance computation. Any metric from scikit-learn
    or scipy.spatial.distance can be used.

    If metric is a callable function, it is called on each
    pair of instances (rows) and the resulting value recorded. The callable
    should take two arrays as input and return one value indicating the
    distance between them. This works for Scipy's metrics, but is less
    efficient than passing the metric name as a string.

    Distance matrices are not supported.

    Valid values for metric are:

    - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
      'manhattan']

    - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
      'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
      'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao',
      'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
      'yule']

    See the documentation for scipy.spatial.distance for details on these
    metrics.

p : int, default=2
    Parameter for the Minkowski metric from
    sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
    equivalent to using manhattan_distance (l1), and euclidean_distance
    (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

metric_params : dict, default=None
    Additional keyword arguments for the metric function.

n_jobs : int, default=1
    The number of parallel jobs to run for neighbors search.
    If ``-1``, then the number of jobs is set to the number of CPU cores.

Examples
--------
>>> from sklearn.cluster import DBSCAN
>>> from sklearn.neighbors import RadiusNeighborsTransformer
>>> from sklearn.pipeline import make_pipeline
>>> estimator = make_pipeline(
...     RadiusNeighborsTransformer(radius=42.0, mode='distance'),
...     DBSCAN(min_samples=30, metric='precomputed'))
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/neighbors/_base.py#L1155)
> `fit(self, X, y=None)`

Fit the model using X as training data

Parameters
----------
X : {array-like, sparse matrix, BallTree, KDTree}
    Training data. If array or matrix, shape [n_samples, n_features],
    or [n_samples, n_samples] if metric='precomputed'.
### `fit_transform`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/neighbors/_graph.py#L449)
> `fit_transform(self, X, y=None)`

Fit to data, then transform it.

Fits transformer to X and y with optional parameters fit_params
and returns a transformed version of X.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Training set.

y : ignored

Returns
-------
Xt : CSR sparse graph, shape (n_samples, n_samples)
    Xt[i, j] is assigned the weight of edge that connects i to j.
    Only the neighbors have an explicit value.
    The diagonal is always explicit.
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
### `radius_neighbors`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/neighbors/_base.py#L829)
> `radius_neighbors(self, X=None, radius=None, return_distance=True, sort_results=False)`

Finds the neighbors within a given radius of a point or points.

Return the indices and distances of each point from the dataset
lying in a ball with size ``radius`` around the points of the query
array. Points lying on the boundary are included in the results.

The result points are *not* necessarily sorted by distance to their
query point.

Parameters
----------
X : array-like, (n_samples, n_features), optional
    The query point or points.
    If not provided, neighbors of each indexed point are returned.
    In this case, the query point is not considered its own neighbor.

radius : float
    Limiting distance of neighbors to return.
    (default is the value passed to the constructor).

return_distance : boolean, optional. Defaults to True.
    If False, distances will not be returned.

sort_results : boolean, optional. Defaults to False.
    If True, the distances and indices will be sorted before being
    returned. If False, the results will not be sorted. If
    return_distance == False, setting sort_results = True will
    result in an error.

    .. versionadded:: 0.22

Returns
-------
neigh_dist : array, shape (n_samples,) of arrays
    Array representing the distances to each point, only present if
    return_distance=True. The distance values are computed according
    to the ``metric`` constructor parameter.

neigh_ind : array, shape (n_samples,) of arrays
    An array of arrays of indices of the approximate nearest points
    from the population matrix that lie within a ball of size
    ``radius`` around the query points.

Examples
--------
In the following example, we construct a NeighborsClassifier
class from an array representing our data set and ask who's
the closest point to [1, 1, 1]:

>>> import numpy as np
>>> samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
>>> from sklearn.neighbors import NearestNeighbors
>>> neigh = NearestNeighbors(radius=1.6)
>>> neigh.fit(samples)
NearestNeighbors(radius=1.6)
>>> rng = neigh.radius_neighbors([[1., 1., 1.]])
>>> print(np.asarray(rng[0][0]))
[1.5 0.5]
>>> print(np.asarray(rng[1][0]))
[1 2]

The first array returned contains the distances to all points which
are closer than 1.6, while the second array returned contains their
indices.  In general, multiple points can be queried at the same time.

Notes
-----
Because the number of neighbors of each point is not necessarily
equal, the results for multiple query points cannot be fit in a
standard data array.
For efficiency, `radius_neighbors` returns arrays of objects, where
each object is a 1D array of indices or distances.
### `radius_neighbors_graph`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/neighbors/_base.py#L1003)
> `radius_neighbors_graph(self, X=None, radius=None, mode='connectivity', sort_results=False)`

Computes the (weighted) graph of Neighbors for points in X

Neighborhoods are restricted the points at a distance lower than
radius.

Parameters
----------
X : array-like of shape (n_samples, n_features), default=None
    The query point or points.
    If not provided, neighbors of each indexed point are returned.
    In this case, the query point is not considered its own neighbor.

radius : float
    Radius of neighborhoods.
    (default is the value passed to the constructor).

mode : {'connectivity', 'distance'}, optional
    Type of returned matrix: 'connectivity' will return the
    connectivity matrix with ones and zeros, in 'distance' the
    edges are Euclidean distance between points.

sort_results : boolean, optional. Defaults to False.
    If True, the distances and indices will be sorted before being
    returned. If False, the results will not be sorted.
    Only used with mode='distance'.

    .. versionadded:: 0.22

Returns
-------
A : sparse graph in CSR format, shape = [n_queries, n_samples_fit]
    n_samples_fit is the number of samples in the fitted data
    A[i, j] is assigned the weight of edge that connects i to j.

Examples
--------
>>> X = [[0], [3], [1]]
>>> from sklearn.neighbors import NearestNeighbors
>>> neigh = NearestNeighbors(radius=1.5)
>>> neigh.fit(X)
NearestNeighbors(radius=1.5)
>>> A = neigh.radius_neighbors_graph(X)
>>> A.toarray()
array([[1., 0., 1.],
       [0., 1., 0.],
       [1., 0., 1.]])

See also
--------
kneighbors_graph
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1301)
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/neighbors/_graph.py#L430)
> `transform(self, X)`

Computes the (weighted) graph of Neighbors for points in X

Parameters
----------
X : array-like of shape (n_samples_transform, n_features)
    Sample data

Returns
-------
Xt : CSR sparse graph of shape (n_samples_transform, n_samples_fit)
    Xt[i, j] is assigned the weight of edge that connects i to j.
    Only the neighbors have an explicit value.
    The diagonal is always explicit.
