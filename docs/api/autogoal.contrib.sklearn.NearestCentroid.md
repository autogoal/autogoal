# `autogoal.contrib.sklearn.NearestCentroid`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1338)
> `NearestCentroid(self)`

Nearest centroid classifier.

Each class is represented by its centroid, with test samples classified to
the class with the nearest centroid.

Read more in the :ref:`User Guide <nearest_centroid_classifier>`.

Parameters
----------
metric : string, or callable
    The metric to use when calculating distance between instances in a
    feature array. If metric is a string or callable, it must be one of
    the options allowed by metrics.pairwise.pairwise_distances for its
    metric parameter.
    The centroids for the samples corresponding to each class is the point
    from which the sum of the distances (according to the metric) of all
    samples that belong to that particular class are minimized.
    If the "manhattan" metric is provided, this centroid is the median and
    for all other metrics, the centroid is now set to be the mean.

shrink_threshold : float, optional (default = None)
    Threshold for shrinking centroids to remove features.

Attributes
----------
centroids_ : array-like of shape (n_classes, n_features)
    Centroid of each class.

classes_ : array of shape (n_classes,)
    The unique classes labels.

Examples
--------
>>> from sklearn.neighbors import NearestCentroid
>>> import numpy as np
>>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
>>> y = np.array([1, 1, 1, 2, 2, 2])
>>> clf = NearestCentroid()
>>> clf.fit(X, y)
NearestCentroid()
>>> print(clf.predict([[-0.8, -1]]))
[1]

See also
--------
sklearn.neighbors.KNeighborsClassifier: nearest neighbors classifier

Notes
-----
When used for text classification with tf-idf vectors, this classifier is
also known as the Rocchio classifier.

References
----------
Tibshirani, R., Hastie, T., Narasimhan, B., & Chu, G. (2002). Diagnosis of
multiple cancer types by shrunken centroids of gene expression. Proceedings
of the National Academy of Sciences of the United States of America,
99(10), 6567-6572. The National Academy of Sciences.
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/neighbors/_nearest_centroid.py#L89)
> `fit(self, X, y)`

Fit the NearestCentroid model according to the given training data.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Training vector, where n_samples is the number of samples and
    n_features is the number of features.
    Note that centroid shrinking cannot be used with sparse matrices.
y : array, shape = [n_samples]
    Target values (integers)
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/neighbors/_nearest_centroid.py#L175)
> `predict(self, X)`

Perform classification on an array of test vectors X.

The predicted class C for each sample in X is returned.

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
C : ndarray of shape (n_samples,)

Notes
-----
If the metric constructor parameter is "precomputed", X is assumed to
be the distance matrix between the data to be predicted and
``self.centroids_``.
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1343)
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

