# `autogoal.contrib.sklearn.Birch`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L63)
> `Birch(self, threshold, branching_factor, n_clusters, compute_labels)`

Implements the Birch clustering algorithm.

It is a memory-efficient, online-learning algorithm provided as an
alternative to :class:`MiniBatchKMeans`. It constructs a tree
data structure with the cluster centroids being read off the leaf.
These can be either the final cluster centroids or can be provided as input
to another clustering algorithm such as :class:`AgglomerativeClustering`.

Read more in the :ref:`User Guide <birch>`.

.. versionadded:: 0.16

Parameters
----------
threshold : float, default=0.5
    The radius of the subcluster obtained by merging a new sample and the
    closest subcluster should be lesser than the threshold. Otherwise a new
    subcluster is started. Setting this value to be very low promotes
    splitting and vice-versa.

branching_factor : int, default=50
    Maximum number of CF subclusters in each node. If a new samples enters
    such that the number of subclusters exceed the branching_factor then
    that node is split into two nodes with the subclusters redistributed
    in each. The parent subcluster of that node is removed and two new
    subclusters are added as parents of the 2 split nodes.

n_clusters : int, instance of sklearn.cluster model, default=3
    Number of clusters after the final clustering step, which treats the
    subclusters from the leaves as new samples.

    - `None` : the final clustering step is not performed and the
      subclusters are returned as they are.

    - :mod:`sklearn.cluster` Estimator : If a model is provided, the model
      is fit treating the subclusters as new samples and the initial data
      is mapped to the label of the closest subcluster.

    - `int` : the model fit is :class:`AgglomerativeClustering` with
      `n_clusters` set to be equal to the int.

compute_labels : bool, default=True
    Whether or not to compute labels for each fit.

copy : bool, default=True
    Whether or not to make a copy of the given data. If set to False,
    the initial data will be overwritten.

Attributes
----------
root_ : _CFNode
    Root of the CFTree.

dummy_leaf_ : _CFNode
    Start pointer to all the leaves.

subcluster_centers_ : ndarray,
    Centroids of all subclusters read directly from the leaves.

subcluster_labels_ : ndarray,
    Labels assigned to the centroids of the subclusters after
    they are clustered globally.

labels_ : ndarray, shape (n_samples,)
    Array of labels assigned to the input data.
    if partial_fit is used instead of fit, they are assigned to the
    last batch of data.

See Also
--------

MiniBatchKMeans
    Alternative  implementation that does incremental updates
    of the centers' positions using mini-batches.

Notes
-----
The tree data structure consists of nodes with each node consisting of
a number of subclusters. The maximum number of subclusters in a node
is determined by the branching factor. Each subcluster maintains a
linear sum, squared sum and the number of samples in that subcluster.
In addition, each subcluster can also have a node as its child, if the
subcluster is not a member of a leaf node.

For a new point entering the root, it is merged with the subcluster closest
to it and the linear sum, squared sum and the number of samples of that
subcluster are updated. This is done recursively till the properties of
the leaf node are updated.

References
----------
* Tian Zhang, Raghu Ramakrishnan, Maron Livny
  BIRCH: An efficient data clustering method for large databases.
  https://www.cs.sfu.ca/CourseCentral/459/han/papers/zhang96.pdf

* Roberto Perdisci
  JBirch - Java implementation of BIRCH clustering algorithm
  https://code.google.com/archive/p/jbirch

Examples
--------
>>> from sklearn.cluster import Birch
>>> X = [[0, 1], [0.3, 1], [-0.3, 1], [0, -1], [0.3, -1], [-0.3, -1]]
>>> brc = Birch(n_clusters=None)
>>> brc.fit(X)
Birch(n_clusters=None)
>>> brc.predict(X)
array([0, 0, 0, 1, 1, 1])
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/cluster/_birch.py#L441)
> `fit(self, X, y=None)`

Build a CF Tree for the input data.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Input data.

y : Ignored
    Not used, present here for API consistency by convention.

Returns
-------
self
    Fitted estimator.
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
### `partial_fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/cluster/_birch.py#L528)
> `partial_fit(self, X=None, y=None)`

Online learning. Prevents rebuilding of CFTree from scratch.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features), None
    Input data. If X is not provided, only the global clustering
    step is done.

y : Ignored
    Not used, present here for API consistency by convention.

Returns
-------
self
    Fitted estimator.
### `predict`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/cluster/_birch.py#L564)
> `predict(self, X)`

Predict data using the ``centroids_`` of subclusters.

Avoid computation of the row norms of X.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Input data.

Returns
-------
labels : ndarray, shape(n_samples)
    Labelled data.
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L80)
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/cluster/_birch.py#L587)
> `transform(self, X)`

Transform X into subcluster centroids dimension.

Each dimension represents the distance from the sample point to each
cluster centroid.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Input data.

Returns
-------
X_trans : {array-like, sparse matrix}, shape (n_samples, n_clusters)
    Transformed data.
