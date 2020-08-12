# `autogoal.contrib.sklearn.FeatureAgglomeration`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L38)
> `FeatureAgglomeration(self, n_clusters, affinity, compute_full_tree, linkage)`

Agglomerate features.

Similar to AgglomerativeClustering, but recursively merges features
instead of samples.

Read more in the :ref:`User Guide <hierarchical_clustering>`.

Parameters
----------
n_clusters : int, default=2
    The number of clusters to find. It must be ``None`` if
    ``distance_threshold`` is not ``None``.

affinity : str or callable, default='euclidean'
    Metric used to compute the linkage. Can be "euclidean", "l1", "l2",
    "manhattan", "cosine", or 'precomputed'.
    If linkage is "ward", only "euclidean" is accepted.

memory : str or object with the joblib.Memory interface, default=None
    Used to cache the output of the computation of the tree.
    By default, no caching is done. If a string is given, it is the
    path to the caching directory.

connectivity : array-like or callable, default=None
    Connectivity matrix. Defines for each feature the neighboring
    features following a given structure of the data.
    This can be a connectivity matrix itself or a callable that transforms
    the data into a connectivity matrix, such as derived from
    kneighbors_graph. Default is None, i.e, the
    hierarchical clustering algorithm is unstructured.

compute_full_tree : 'auto' or bool, optional, default='auto'
    Stop early the construction of the tree at n_clusters. This is useful
    to decrease computation time if the number of clusters is not small
    compared to the number of features. This option is useful only when
    specifying a connectivity matrix. Note also that when varying the
    number of clusters and using caching, it may be advantageous to compute
    the full tree. It must be ``True`` if ``distance_threshold`` is not
    ``None``. By default `compute_full_tree` is "auto", which is equivalent
    to `True` when `distance_threshold` is not `None` or that `n_clusters`
    is inferior to the maximum between 100 or `0.02 * n_samples`.
    Otherwise, "auto" is equivalent to `False`.

linkage : {'ward', 'complete', 'average', 'single'}, default='ward'
    Which linkage criterion to use. The linkage criterion determines which
    distance to use between sets of features. The algorithm will merge
    the pairs of cluster that minimize this criterion.

    - ward minimizes the variance of the clusters being merged.
    - average uses the average of the distances of each feature of
      the two sets.
    - complete or maximum linkage uses the maximum distances between
      all features of the two sets.
    - single uses the minimum of the distances between all observations
      of the two sets.

pooling_func : callable, default=np.mean
    This combines the values of agglomerated features into a single
    value, and should accept an array of shape [M, N] and the keyword
    argument `axis=1`, and reduce it to an array of size [M].

distance_threshold : float, default=None
    The linkage distance threshold above which, clusters will not be
    merged. If not ``None``, ``n_clusters`` must be ``None`` and
    ``compute_full_tree`` must be ``True``.

    .. versionadded:: 0.21

Attributes
----------
n_clusters_ : int
    The number of clusters found by the algorithm. If
    ``distance_threshold=None``, it will be equal to the given
    ``n_clusters``.

labels_ : array-like of (n_features,)
    cluster labels for each feature.

n_leaves_ : int
    Number of leaves in the hierarchical tree.

n_connected_components_ : int
    The estimated number of connected components in the graph.

children_ : array-like of shape (n_nodes-1, 2)
    The children of each non-leaf node. Values less than `n_features`
    correspond to leaves of the tree which are the original samples.
    A node `i` greater than or equal to `n_features` is a non-leaf
    node and has children `children_[i - n_features]`. Alternatively
    at the i-th iteration, children[i][0] and children[i][1]
    are merged to form node `n_features + i`

distances_ : array-like of shape (n_nodes-1,)
    Distances between nodes in the corresponding place in `children_`.
    Only computed if distance_threshold is not None.

Examples
--------
>>> import numpy as np
>>> from sklearn import datasets, cluster
>>> digits = datasets.load_digits()
>>> images = digits.images
>>> X = np.reshape(images, (len(images), -1))
>>> agglo = cluster.FeatureAgglomeration(n_clusters=32)
>>> agglo.fit(X)
FeatureAgglomeration(n_clusters=32)
>>> X_reduced = agglo.transform(X)
>>> X_reduced.shape
(1797, 32)
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/cluster/_agglomerative.py#L1028)
> `fit(self, X, y=None, **params)`

Fit the hierarchical clustering on the data

Parameters
----------
X : array-like of shape (n_samples, n_features)
    The data

y : Ignored

Returns
-------
self
### `fit_transform`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/base.py#L544)
> `fit_transform(self, X, y=None, **fit_params)`

Fit to data, then transform it.

Fits transformer to X and y with optional parameters fit_params
and returns a transformed version of X.

Parameters
----------
X : numpy array of shape [n_samples, n_features]
    Training set.

y : numpy array of shape [n_samples]
    Target values.

**fit_params : dict
    Additional fit parameters.

Returns
-------
X_new : numpy array of shape [n_samples, n_features_new]
    Transformed array.
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/cluster/_feature_agglomeration.py#L57)
> `inverse_transform(self, Xred)`

Inverse the transformation.
Return a vector of size nb_features with the values of Xred assigned
to each group of features

Parameters
----------
Xred : array-like of shape (n_samples, n_clusters) or (n_clusters,)
    The values to be assigned to each cluster of samples

Returns
-------
X : array, shape=[n_samples, n_features] or [n_features]
    A vector of size n_samples with the values of Xred assigned to
    each of the cluster of samples.
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L55)
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/cluster/_feature_agglomeration.py#L24)
> `transform(self, X)`

Transform a new matrix using the built clustering

Parameters
----------
X : array-like of shape (n_samples, n_features) or (n_samples,)
    A M by N array of M observations in N dimensions or a length
    M array of M one-dimensional observations.

Returns
-------
Y : array, shape = [n_samples, n_clusters] or [n_clusters]
    The pooled values for each feature cluster.
