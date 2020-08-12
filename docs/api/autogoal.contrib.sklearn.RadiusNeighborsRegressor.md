# `autogoal.contrib.sklearn.RadiusNeighborsRegressor`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1386)
> `RadiusNeighborsRegressor(self, radius, weights, algorithm, leaf_size, p)`

Regression based on neighbors within a fixed radius.

The target is predicted by local interpolation of the targets
associated of the nearest neighbors in the training set.

Read more in the :ref:`User Guide <regression>`.

.. versionadded:: 0.9

Parameters
----------
radius : float, optional (default = 1.0)
    Range of parameter space to use by default for :meth:`radius_neighbors`
    queries.

weights : str or callable
    weight function used in prediction.  Possible values:

    - 'uniform' : uniform weights.  All points in each neighborhood
      are weighted equally.
    - 'distance' : weight points by the inverse of their distance.
      in this case, closer neighbors of a query point will have a
      greater influence than neighbors which are further away.
    - [callable] : a user-defined function which accepts an
      array of distances, and returns an array of the same shape
      containing the weights.

    Uniform weights are used by default.

algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
    Algorithm used to compute the nearest neighbors:

    - 'ball_tree' will use :class:`BallTree`
    - 'kd_tree' will use :class:`KDTree`
    - 'brute' will use a brute-force search.
    - 'auto' will attempt to decide the most appropriate algorithm
      based on the values passed to :meth:`fit` method.

    Note: fitting on sparse input will override the setting of
    this parameter, using brute force.

leaf_size : int, optional (default = 30)
    Leaf size passed to BallTree or KDTree.  This can affect the
    speed of the construction and query, as well as the memory
    required to store the tree.  The optimal value depends on the
    nature of the problem.

p : integer, optional (default = 2)
    Power parameter for the Minkowski metric. When p = 1, this is
    equivalent to using manhattan_distance (l1), and euclidean_distance
    (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

metric : string or callable, default 'minkowski'
    the distance metric to use for the tree.  The default metric is
    minkowski, and with p=2 is equivalent to the standard Euclidean
    metric. See the documentation of the DistanceMetric class for a
    list of available metrics.
    If metric is "precomputed", X is assumed to be a distance matrix and
    must be square during fit. X may be a :term:`Glossary <sparse graph>`,
    in which case only "nonzero" elements may be considered neighbors.

metric_params : dict, optional (default = None)
    Additional keyword arguments for the metric function.

n_jobs : int or None, optional (default=None)
    The number of parallel jobs to run for neighbors search.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

Attributes
----------
effective_metric_ : string or callable
    The distance metric to use. It will be same as the `metric` parameter
    or a synonym of it, e.g. 'euclidean' if the `metric` parameter set to
    'minkowski' and `p` parameter set to 2.

effective_metric_params_ : dict
    Additional keyword arguments for the metric function. For most metrics
    will be same with `metric_params` parameter, but may also contain the
    `p` parameter value if the `effective_metric_` attribute is set to
    'minkowski'.

Examples
--------
>>> X = [[0], [1], [2], [3]]
>>> y = [0, 0, 1, 1]
>>> from sklearn.neighbors import RadiusNeighborsRegressor
>>> neigh = RadiusNeighborsRegressor(radius=1.0)
>>> neigh.fit(X, y)
RadiusNeighborsRegressor(...)
>>> print(neigh.predict([[1.5]]))
[0.5]

See also
--------
NearestNeighbors
KNeighborsRegressor
KNeighborsClassifier
RadiusNeighborsClassifier

Notes
-----
See :ref:`Nearest Neighbors <neighbors>` in the online documentation
for a discussion of the choice of ``algorithm`` and ``leaf_size``.

https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/neighbors/_base.py#L1092)
> `fit(self, X, y)`

Fit the model using X as training data and y as target values

Parameters
----------
X : {array-like, sparse matrix, BallTree, KDTree}
    Training data. If array or matrix, shape [n_samples, n_features],
    or [n_samples, n_samples] if metric='precomputed'.

y : {array-like, sparse matrix}
    Target values, array of float values, shape = [n_samples]
     or [n_samples, n_outputs]
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
### `predict`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/neighbors/_regression.py#L322)
> `predict(self, X)`

Predict the target for the provided data

Parameters
----------
X : array-like, shape (n_queries, n_features),                 or (n_queries, n_indexed) if metric == 'precomputed'
    Test samples.

Returns
-------
y : array of float, shape = [n_queries] or [n_queries, n_outputs]
    Target values
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

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1405)
> `run(self, input)`

### `score`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/base.py#L376)
> `score(self, X, y, sample_weight=None)`

Return the coefficient of determination R^2 of the prediction.

The coefficient R^2 is defined as (1 - u/v), where u is the residual
sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
sum of squares ((y_true - y_true.mean()) ** 2).sum().
The best possible score is 1.0 and it can be negative (because the
model can be arbitrarily worse). A constant model that always
predicts the expected value of y, disregarding the input features,
would get a R^2 score of 0.0.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Test samples. For some estimators this may be a
    precomputed kernel matrix or a list of generic objects instead,
    shape = (n_samples, n_samples_fitted),
    where n_samples_fitted is the number of
    samples used in the fitting for the estimator.

y : array-like of shape (n_samples,) or (n_samples, n_outputs)
    True values for X.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

Returns
-------
score : float
    R^2 of self.predict(X) wrt. y.

Notes
-----
The R2 score used when calling ``score`` on a regressor will use
``multioutput='uniform_average'`` from version 0.23 to keep consistent
with :func:`~sklearn.metrics.r2_score`. This will influence the
``score`` method of all the multioutput regressors (except for
:class:`~sklearn.multioutput.MultiOutputRegressor`). To specify the
default value manually and avoid the warning, please either call
:func:`~sklearn.metrics.r2_score` directly or make a custom scorer with
:func:`~sklearn.metrics.make_scorer` (the built-in scorer ``'r2'`` uses
``multioutput='uniform_average'``).
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

