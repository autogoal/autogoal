# `autogoal.contrib.sklearn.ComplementNB`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1175)
> `ComplementNB(self, fit_prior, norm)`

The Complement Naive Bayes classifier described in Rennie et al. (2003).

The Complement Naive Bayes classifier was designed to correct the "severe
assumptions" made by the standard Multinomial Naive Bayes classifier. It is
particularly suited for imbalanced data sets.

Read more in the :ref:`User Guide <complement_naive_bayes>`.

Parameters
----------
alpha : float, optional (default=1.0)
    Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).

fit_prior : boolean, optional (default=True)
    Only used in edge case with a single class in the training set.

class_prior : array-like, size (n_classes,), optional (default=None)
    Prior probabilities of the classes. Not used.

norm : boolean, optional (default=False)
    Whether or not a second normalization of the weights is performed. The
    default behavior mirrors the implementations found in Mahout and Weka,
    which do not follow the full algorithm described in Table 9 of the
    paper.

Attributes
----------
class_count_ : array, shape (n_classes,)
    Number of samples encountered for each class during fitting. This
    value is weighted by the sample weight when provided.

class_log_prior_ : array, shape (n_classes, )
    Smoothed empirical log probability for each class. Only used in edge
    case with a single class in the training set.

classes_ : array, shape (n_classes,)
    Class labels known to the classifier

feature_all_ : array, shape (n_features,)
    Number of samples encountered for each feature during fitting. This
    value is weighted by the sample weight when provided.

feature_count_ : array, shape (n_classes, n_features)
    Number of samples encountered for each (class, feature) during fitting.
    This value is weighted by the sample weight when provided.

feature_log_prob_ : array, shape (n_classes, n_features)
    Empirical weights for class complements.

n_features_ : int
    Number of features of each sample.

Examples
--------
>>> import numpy as np
>>> rng = np.random.RandomState(1)
>>> X = rng.randint(5, size=(6, 100))
>>> y = np.array([1, 2, 3, 4, 5, 6])
>>> from sklearn.naive_bayes import ComplementNB
>>> clf = ComplementNB()
>>> clf.fit(X, y)
ComplementNB()
>>> print(clf.predict(X[2:3]))
[3]

References
----------
Rennie, J. D., Shih, L., Teevan, J., & Karger, D. R. (2003).
Tackling the poor assumptions of naive bayes text classifiers. In ICML
(Vol. 3, pp. 616-623).
https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/naive_bayes.py#L590)
> `fit(self, X, y, sample_weight=None)`

Fit Naive Bayes classifier according to X, y

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

y : array-like of shape (n_samples,)
    Target values.

sample_weight : array-like of shape (n_samples,), default=None
    Weights applied to individual samples (1. for unweighted).

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
### `partial_fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/naive_bayes.py#L511)
> `partial_fit(self, X, y, classes=None, sample_weight=None)`

Incremental fit on a batch of samples.

This method is expected to be called several times consecutively
on different chunks of a dataset so as to implement out-of-core
or online learning.

This is especially useful when the whole dataset is too big to fit in
memory at once.

This method has some performance overhead hence it is better to call
partial_fit on chunks of data that are as large as possible
(as long as fitting in the memory budget) to hide the overhead.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

y : array-like of shape (n_samples,)
    Target values.

classes : array-like of shape (n_classes) (default=None)
    List of all the classes that can possibly appear in the y vector.

    Must be provided at the first call to partial_fit, can be omitted
    in subsequent calls.

sample_weight : array-like of shape (n_samples,), default=None
    Weights applied to individual samples (1. for unweighted).

Returns
-------
self : object
### `predict`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/naive_bayes.py#L62)
> `predict(self, X)`

Perform classification on an array of test vectors X.

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
C : ndarray of shape (n_samples,)
    Predicted target values for X
### `predict_log_proba`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/naive_bayes.py#L80)
> `predict_log_proba(self, X)`

Return log-probability estimates for the test vector X.

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
C : array-like of shape (n_samples, n_classes)
    Returns the log-probability of the samples for each class in
    the model. The columns correspond to the classes in sorted
    order, as they appear in the attribute :term:`classes_`.
### `predict_proba`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/naive_bayes.py#L102)
> `predict_proba(self, X)`

Return probability estimates for the test vector X.

Parameters
----------
X : array-like of shape (n_samples, n_features)

Returns
-------
C : array-like of shape (n_samples, n_classes)
    Returns the probability of the samples for each class in
    the model. The columns correspond to the classes in sorted
    order, as they appear in the attribute :term:`classes_`.
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1180)
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

