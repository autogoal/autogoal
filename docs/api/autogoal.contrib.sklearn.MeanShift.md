# `autogoal.contrib.sklearn.MeanShift`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L146)
> `MeanShift(self, bin_seeding, cluster_all)`

Mean shift clustering using a flat kernel.

Mean shift clustering aims to discover "blobs" in a smooth density of
samples. It is a centroid-based algorithm, which works by updating
candidates for centroids to be the mean of the points within a given
region. These candidates are then filtered in a post-processing stage to
eliminate near-duplicates to form the final set of centroids.

Seeding is performed using a binning technique for scalability.

Read more in the :ref:`User Guide <mean_shift>`.

Parameters
----------
bandwidth : float, optional
    Bandwidth used in the RBF kernel.

    If not given, the bandwidth is estimated using
    sklearn.cluster.estimate_bandwidth; see the documentation for that
    function for hints on scalability (see also the Notes, below).

seeds : array, shape=[n_samples, n_features], optional
    Seeds used to initialize kernels. If not set,
    the seeds are calculated by clustering.get_bin_seeds
    with bandwidth as the grid size and default values for
    other parameters.

bin_seeding : boolean, optional
    If true, initial kernel locations are not locations of all
    points, but rather the location of the discretized version of
    points, where points are binned onto a grid whose coarseness
    corresponds to the bandwidth. Setting this option to True will speed
    up the algorithm because fewer seeds will be initialized.
    default value: False
    Ignored if seeds argument is not None.

min_bin_freq : int, optional
   To speed up the algorithm, accept only those bins with at least
   min_bin_freq points as seeds. If not defined, set to 1.

cluster_all : boolean, default True
    If true, then all points are clustered, even those orphans that are
    not within any kernel. Orphans are assigned to the nearest kernel.
    If false, then orphans are given cluster label -1.

n_jobs : int or None, optional (default=None)
    The number of jobs to use for the computation. This works by computing
    each of the n_init runs in parallel.

    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

max_iter : int, default=300
    Maximum number of iterations, per seed point before the clustering
    operation terminates (for that seed point), if has not converged yet.

    .. versionadded:: 0.22

Attributes
----------
cluster_centers_ : array, [n_clusters, n_features]
    Coordinates of cluster centers.

labels_ :
    Labels of each point.

n_iter_ : int
    Maximum number of iterations performed on each seed.

    .. versionadded:: 0.22

Examples
--------
>>> from sklearn.cluster import MeanShift
>>> import numpy as np
>>> X = np.array([[1, 1], [2, 1], [1, 0],
...               [4, 7], [3, 5], [3, 6]])
>>> clustering = MeanShift(bandwidth=2).fit(X)
>>> clustering.labels_
array([1, 1, 1, 0, 0, 0])
>>> clustering.predict([[0, 0], [5, 5]])
array([1, 0])
>>> clustering
MeanShift(bandwidth=2)

Notes
-----

Scalability:

Because this implementation uses a flat kernel and
a Ball Tree to look up members of each kernel, the complexity will tend
towards O(T*n*log(n)) in lower dimensions, with n the number of samples
and T the number of points. In higher dimensions the complexity will
tend towards O(T*n^2).

Scalability can be boosted by using fewer seeds, for example by using
a higher value of min_bin_freq in the get_bin_seeds function.

Note that the estimate_bandwidth function is much less scalable than the
mean shift algorithm and will be the bottleneck if it is used.

References
----------

Dorin Comaniciu and Peter Meer, "Mean Shift: A robust approach toward
feature space analysis". IEEE Transactions on Pattern Analysis and
Machine Intelligence. 2002. pp. 603-619.
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/cluster/_mean_shift.py#L359)
> `fit(self, X, y=None)`

Perform clustering.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Samples to cluster.

y : Ignored
### `fit_predict`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/base.py#L443)
> `fit_predict(self, X, y=None)`

Perform clustering on X and returns cluster labels.

Parameters
----------
X : ndarray, shape (n_samples, n_features)
    Input data.

y : Ignored
    Not used, present for API consistency by convention.

Returns
-------
labels : ndarray, shape (n_samples,)
    Cluster labels.
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/cluster/_mean_shift.py#L445)
> `predict(self, X)`

Predict the closest cluster each sample in X belongs to.

Parameters
----------
X : {array-like, sparse matrix}, shape=[n_samples, n_features]
    New data to predict.

Returns
-------
labels : array, shape [n_samples,]
    Index of the cluster each sample belongs to.
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L151)
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

