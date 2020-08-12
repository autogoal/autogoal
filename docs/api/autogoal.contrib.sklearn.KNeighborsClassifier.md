# `autogoal.contrib.sklearn.KNeighborsClassifier`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1222)
> `KNeighborsClassifier(self, n_neighbors, weights, algorithm, leaf_size, p, metric)`

Classifier implementing the k-nearest neighbors vote.

Read more in the :ref:`User Guide <classification>`.

Parameters
----------
n_neighbors : int, optional (default = 5)
    Number of neighbors to use by default for :meth:`kneighbors` queries.

weights : str or callable, optional (default = 'uniform')
    weight function used in prediction.  Possible values:

    - 'uniform' : uniform weights.  All points in each neighborhood
      are weighted equally.
    - 'distance' : weight points by the inverse of their distance.
      in this case, closer neighbors of a query point will have a
      greater influence than neighbors which are further away.
    - [callable] : a user-defined function which accepts an
      array of distances, and returns an array of the same shape
      containing the weights.

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
    Doesn't affect :meth:`fit` method.

Attributes
----------
classes_ : array of shape (n_classes,)
    Class labels known to the classifier

effective_metric_ : string or callble
    The distance metric used. It will be same as the `metric` parameter
    or a synonym of it, e.g. 'euclidean' if the `metric` parameter set to
    'minkowski' and `p` parameter set to 2.

effective_metric_params_ : dict
    Additional keyword arguments for the metric function. For most metrics
    will be same with `metric_params` parameter, but may also contain the
    `p` parameter value if the `effective_metric_` attribute is set to
    'minkowski'.

outputs_2d_ : bool
    False when `y`'s shape is (n_samples, ) or (n_samples, 1) during fit
    otherwise True.

Examples
--------
>>> X = [[0], [1], [2], [3]]
>>> y = [0, 0, 1, 1]
>>> from sklearn.neighbors import KNeighborsClassifier
>>> neigh = KNeighborsClassifier(n_neighbors=3)
>>> neigh.fit(X, y)
KNeighborsClassifier(...)
>>> print(neigh.predict([[1.1]]))
[0]
>>> print(neigh.predict_proba([[0.9]]))
[[0.66666667 0.33333333]]

See also
--------
RadiusNeighborsClassifier
KNeighborsRegressor
RadiusNeighborsRegressor
NearestNeighbors

Notes
-----
See :ref:`Nearest Neighbors <neighbors>` in the online documentation
for a discussion of the choice of ``algorithm`` and ``leaf_size``.

.. warning::

   Regarding the Nearest Neighbors algorithms, if it is found that two
   neighbors, neighbor `k+1` and `k`, have identical distances
   but different labels, the results will depend on the ordering of the
   training data.

https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/neighbors/_base.py#L1112)
> `fit(self, X, y)`

Fit the model using X as training data and y as target values

Parameters
----------
X : {array-like, sparse matrix, BallTree, KDTree}
    Training data. If array or matrix, shape [n_samples, n_features],
    or [n_samples, n_samples] if metric='precomputed'.

y : {array-like, sparse matrix}
    Target values of shape = [n_samples] or [n_samples, n_outputs]
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
### `kneighbors`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/neighbors/_base.py#L532)
> `kneighbors(self, X=None, n_neighbors=None, return_distance=True)`

Finds the K-neighbors of a point.
Returns indices of and distances to the neighbors of each point.

Parameters
----------
X : array-like, shape (n_queries, n_features),                 or (n_queries, n_indexed) if metric == 'precomputed'
    The query point or points.
    If not provided, neighbors of each indexed point are returned.
    In this case, the query point is not considered its own neighbor.

n_neighbors : int
    Number of neighbors to get (default is the value
    passed to the constructor).

return_distance : boolean, optional. Defaults to True.
    If False, distances will not be returned

Returns
-------
neigh_dist : array, shape (n_queries, n_neighbors)
    Array representing the lengths to points, only present if
    return_distance=True

neigh_ind : array, shape (n_queries, n_neighbors)
    Indices of the nearest points in the population matrix.

Examples
--------
In the following example, we construct a NearestNeighbors
class from an array representing our data set and ask who's
the closest point to [1,1,1]

>>> samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
>>> from sklearn.neighbors import NearestNeighbors
>>> neigh = NearestNeighbors(n_neighbors=1)
>>> neigh.fit(samples)
NearestNeighbors(n_neighbors=1)
>>> print(neigh.kneighbors([[1., 1., 1.]]))
(array([[0.5]]), array([[2]]))

As you can see, it returns [[0.5]], and [[2]], which means that the
element is at distance 0.5 and is the third element of samples
(indexes start at 0). You can also query for multiple points:

>>> X = [[0., 1., 0.], [1., 0., 1.]]
>>> neigh.kneighbors(X, return_distance=False)
array([[1],
       [2]]...)
### `kneighbors_graph`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/neighbors/_base.py#L706)
> `kneighbors_graph(self, X=None, n_neighbors=None, mode='connectivity')`

Computes the (weighted) graph of k-Neighbors for points in X

Parameters
----------
X : array-like, shape (n_queries, n_features),                 or (n_queries, n_indexed) if metric == 'precomputed'
    The query point or points.
    If not provided, neighbors of each indexed point are returned.
    In this case, the query point is not considered its own neighbor.

n_neighbors : int
    Number of neighbors for each sample.
    (default is value passed to the constructor).

mode : {'connectivity', 'distance'}, optional
    Type of returned matrix: 'connectivity' will return the
    connectivity matrix with ones and zeros, in 'distance' the
    edges are Euclidean distance between points.

Returns
-------
A : sparse graph in CSR format, shape = [n_queries, n_samples_fit]
    n_samples_fit is the number of samples in the fitted data
    A[i, j] is assigned the weight of edge that connects i to j.

Examples
--------
>>> X = [[0], [3], [1]]
>>> from sklearn.neighbors import NearestNeighbors
>>> neigh = NearestNeighbors(n_neighbors=2)
>>> neigh.fit(X)
NearestNeighbors(n_neighbors=2)
>>> A = neigh.kneighbors_graph(X)
>>> A.toarray()
array([[1., 0., 1.],
       [0., 1., 1.],
       [1., 0., 1.]])

See also
--------
NearestNeighbors.radius_neighbors_graph
### `predict`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/neighbors/_classification.py#L157)
> `predict(self, X)`

Predict the class labels for the provided data.

Parameters
----------
X : array-like, shape (n_queries, n_features),                 or (n_queries, n_indexed) if metric == 'precomputed'
    Test samples.

Returns
-------
y : array of shape [n_queries] or [n_queries, n_outputs]
    Class labels for each data sample.
### `predict_proba`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/neighbors/_classification.py#L199)
> `predict_proba(self, X)`

Return probability estimates for the test data X.

Parameters
----------
X : array-like, shape (n_queries, n_features),                 or (n_queries, n_indexed) if metric == 'precomputed'
    Test samples.

Returns
-------
p : array of shape = [n_queries, n_classes], or a list of n_outputs
    of such arrays if n_outputs > 1.
    The class probabilities of the input samples. Classes are ordered
    by lexicographic order.
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1243)
> `run(self, input)`

### `score`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/base.py#L344)
> `score(self, X, y, sample_weight=None)`

Return the mean accuracy on the given test data and labels.

In multi-label classification, this is the subset accuracy
which is a harsh metric since you require for each sample that
each label set be correctly predicted.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Test samples.

y : array-like of shape (n_samples,) or (n_samples, n_outputs)
    True labels for X.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

Returns
-------
score : float
    Mean accuracy of self.predict(X) wrt. y.
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

