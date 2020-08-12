# `autogoal.contrib.sklearn.LocalOutlierFactor`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1309)
> `LocalOutlierFactor(self, n_neighbors, algorithm, leaf_size, p, contamination, novelty)`

Unsupervised Outlier Detection using Local Outlier Factor (LOF)

The anomaly score of each sample is called Local Outlier Factor.
It measures the local deviation of density of a given sample with
respect to its neighbors.
It is local in that the anomaly score depends on how isolated the object
is with respect to the surrounding neighborhood.
More precisely, locality is given by k-nearest neighbors, whose distance
is used to estimate the local density.
By comparing the local density of a sample to the local densities of
its neighbors, one can identify samples that have a substantially lower
density than their neighbors. These are considered outliers.

.. versionadded:: 0.19

Parameters
----------
n_neighbors : int, optional (default=20)
    Number of neighbors to use by default for :meth:`kneighbors` queries.
    If n_neighbors is larger than the number of samples provided,
    all samples will be used.

algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
    Algorithm used to compute the nearest neighbors:

    - 'ball_tree' will use :class:`BallTree`
    - 'kd_tree' will use :class:`KDTree`
    - 'brute' will use a brute-force search.
    - 'auto' will attempt to decide the most appropriate algorithm
      based on the values passed to :meth:`fit` method.

    Note: fitting on sparse input will override the setting of
    this parameter, using brute force.

leaf_size : int, optional (default=30)
    Leaf size passed to :class:`BallTree` or :class:`KDTree`. This can
    affect the speed of the construction and query, as well as the memory
    required to store the tree. The optimal value depends on the
    nature of the problem.

metric : string or callable, default 'minkowski'
    metric used for the distance computation. Any metric from scikit-learn
    or scipy.spatial.distance can be used.

    If metric is "precomputed", X is assumed to be a distance matrix and
    must be square. X may be a sparse matrix, in which case only "nonzero"
    elements may be considered neighbors.

    If metric is a callable function, it is called on each
    pair of instances (rows) and the resulting value recorded. The callable
    should take two arrays as input and return one value indicating the
    distance between them. This works for Scipy's metrics, but is less
    efficient than passing the metric name as a string.

    Valid values for metric are:

    - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
      'manhattan']

    - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
      'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
      'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao',
      'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
      'yule']

    See the documentation for scipy.spatial.distance for details on these
    metrics:
    https://docs.scipy.org/doc/scipy/reference/spatial.distance.html

p : integer, optional (default=2)
    Parameter for the Minkowski metric from
    :func:`sklearn.metrics.pairwise.pairwise_distances`. When p = 1, this
    is equivalent to using manhattan_distance (l1), and euclidean_distance
    (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

metric_params : dict, optional (default=None)
    Additional keyword arguments for the metric function.

contamination : 'auto' or float, optional (default='auto')
    The amount of contamination of the data set, i.e. the proportion
    of outliers in the data set. When fitting this is used to define the
    threshold on the scores of the samples.

    - if 'auto', the threshold is determined as in the
      original paper,
    - if a float, the contamination should be in the range [0, 0.5].

    .. versionchanged:: 0.22
       The default value of ``contamination`` changed from 0.1
       to ``'auto'``.

novelty : boolean, default False
    By default, LocalOutlierFactor is only meant to be used for outlier
    detection (novelty=False). Set novelty to True if you want to use
    LocalOutlierFactor for novelty detection. In this case be aware that
    that you should only use predict, decision_function and score_samples
    on new unseen data and not on the training set.

n_jobs : int or None, optional (default=None)
    The number of parallel jobs to run for neighbors search.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

Attributes
----------
negative_outlier_factor_ : numpy array, shape (n_samples,)
    The opposite LOF of the training samples. The higher, the more normal.
    Inliers tend to have a LOF score close to 1 (``negative_outlier_factor_``
    close to -1), while outliers tend to have a larger LOF score.

    The local outlier factor (LOF) of a sample captures its
    supposed 'degree of abnormality'.
    It is the average of the ratio of the local reachability density of
    a sample and those of its k-nearest neighbors.

n_neighbors_ : integer
    The actual number of neighbors used for :meth:`kneighbors` queries.

offset_ : float
    Offset used to obtain binary labels from the raw scores.
    Observations having a negative_outlier_factor smaller than `offset_`
    are detected as abnormal.
    The offset is set to -1.5 (inliers score around -1), except when a
    contamination parameter different than "auto" is provided. In that
    case, the offset is defined in such a way we obtain the expected
    number of outliers in training.

Examples
--------
>>> import numpy as np
>>> from sklearn.neighbors import LocalOutlierFactor
>>> X = [[-1.1], [0.2], [101.1], [0.3]]
>>> clf = LocalOutlierFactor(n_neighbors=2)
>>> clf.fit_predict(X)
array([ 1,  1, -1,  1])
>>> clf.negative_outlier_factor_
array([ -0.9821...,  -1.0370..., -73.3697...,  -0.9821...])

References
----------
.. [1] Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J. (2000, May).
       LOF: identifying density-based local outliers. In ACM sigmod record.
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/neighbors/_lof.py#L231)
> `fit(self, X, y=None)`

Fit the model using X as training data.

Parameters
----------
X : {array-like, sparse matrix, BallTree, KDTree}
    Training data. If array or matrix, shape [n_samples, n_features],
    or [n_samples, n_samples] if metric='precomputed'.

y : Ignored
    not used, present for API consistency by convention.

Returns
-------
self : object
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
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1330)
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

