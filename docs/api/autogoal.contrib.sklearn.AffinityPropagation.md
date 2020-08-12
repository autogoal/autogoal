# `autogoal.contrib.sklearn.AffinityPropagation`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L17)
> `AffinityPropagation(self, convergence_iter, affinity)`

Perform Affinity Propagation Clustering of data.

Read more in the :ref:`User Guide <affinity_propagation>`.

Parameters
----------
damping : float, default=0.5
    Damping factor (between 0.5 and 1) is the extent to
    which the current value is maintained relative to
    incoming values (weighted 1 - damping). This in order
    to avoid numerical oscillations when updating these
    values (messages).

max_iter : int, default=200
    Maximum number of iterations.

convergence_iter : int, default=15
    Number of iterations with no change in the number
    of estimated clusters that stops the convergence.

copy : bool, default=True
    Make a copy of input data.

preference : array-like of shape (n_samples,) or float, default=None
    Preferences for each point - points with larger values of
    preferences are more likely to be chosen as exemplars. The number
    of exemplars, ie of clusters, is influenced by the input
    preferences value. If the preferences are not passed as arguments,
    they will be set to the median of the input similarities.

affinity : {'euclidean', 'precomputed'}, default='euclidean'
    Which affinity to use. At the moment 'precomputed' and
    ``euclidean`` are supported. 'euclidean' uses the
    negative squared euclidean distance between points.

verbose : bool, default=False
    Whether to be verbose.


Attributes
----------
cluster_centers_indices_ : ndarray of shape (n_clusters,)
    Indices of cluster centers

cluster_centers_ : ndarray of shape (n_clusters, n_features)
    Cluster centers (if affinity != ``precomputed``).

labels_ : ndarray of shape (n_samples,)
    Labels of each point

affinity_matrix_ : ndarray of shape (n_samples, n_samples)
    Stores the affinity matrix used in ``fit``.

n_iter_ : int
    Number of iterations taken to converge.

Examples
--------
>>> from sklearn.cluster import AffinityPropagation
>>> import numpy as np
>>> X = np.array([[1, 2], [1, 4], [1, 0],
...               [4, 2], [4, 4], [4, 0]])
>>> clustering = AffinityPropagation().fit(X)
>>> clustering
AffinityPropagation()
>>> clustering.labels_
array([0, 0, 0, 1, 1, 1])
>>> clustering.predict([[0, 0], [4, 4]])
array([0, 1])
>>> clustering.cluster_centers_
array([[1, 2],
       [4, 2]])

Notes
-----
For an example, see :ref:`examples/cluster/plot_affinity_propagation.py
<sphx_glr_auto_examples_cluster_plot_affinity_propagation.py>`.

The algorithmic complexity of affinity propagation is quadratic
in the number of points.

When ``fit`` does not converge, ``cluster_centers_`` becomes an empty
array and all training samples will be labelled as ``-1``. In addition,
``predict`` will then label every sample as ``-1``.

When all training samples have equal similarities and equal preferences,
the assignment of cluster centers and labels depends on the preference.
If the preference is smaller than the similarities, ``fit`` will result in
a single cluster center and label ``0`` for every sample. Otherwise, every
training sample becomes its own cluster center and is assigned a unique
label.

References
----------

Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages
Between Data Points", Science Feb. 2007
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/cluster/_affinity_propagation.py#L354)
> `fit(self, X, y=None)`

Fit the clustering from features, or affinity matrix.

Parameters
----------
X : array-like or sparse matrix, shape (n_samples, n_features), or             array-like, shape (n_samples, n_samples)
    Training instances to cluster, or similarities / affinities between
    instances if ``affinity='precomputed'``. If a sparse feature matrix
    is provided, it will be converted into a sparse ``csr_matrix``.

y : Ignored
    Not used, present here for API consistency by convention.

Returns
-------
self
### `fit_predict`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/cluster/_affinity_propagation.py#L426)
> `fit_predict(self, X, y=None)`

Fit the clustering from features or affinity matrix, and return
cluster labels.

Parameters
----------
X : array-like or sparse matrix, shape (n_samples, n_features), or             array-like, shape (n_samples, n_samples)
    Training instances to cluster, or similarities / affinities between
    instances if ``affinity='precomputed'``. If a sparse feature matrix
    is provided, it will be converted into a sparse ``csr_matrix``.

y : Ignored
    Not used, present here for API consistency by convention.

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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/cluster/_affinity_propagation.py#L398)
> `predict(self, X)`

Predict the closest cluster each sample in X belongs to.

Parameters
----------
X : array-like or sparse matrix, shape (n_samples, n_features)
    New data to predict. If a sparse matrix is provided, it will be
    converted into a sparse ``csr_matrix``.

Returns
-------
labels : ndarray, shape (n_samples,)
    Cluster labels.
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L28)
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

