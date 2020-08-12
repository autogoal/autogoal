# `autogoal.contrib.sklearn.GaussianNB`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1190)
> `GaussianNB(self)`

Gaussian Naive Bayes (GaussianNB)

Can perform online updates to model parameters via :meth:`partial_fit`.
For details on algorithm used to update feature means and variance online,
see Stanford CS tech report STAN-CS-79-773 by Chan, Golub, and LeVeque:

    http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf

Read more in the :ref:`User Guide <gaussian_naive_bayes>`.

Parameters
----------
priors : array-like, shape (n_classes,)
    Prior probabilities of the classes. If specified the priors are not
    adjusted according to the data.

var_smoothing : float, optional (default=1e-9)
    Portion of the largest variance of all features that is added to
    variances for calculation stability.

Attributes
----------
class_count_ : array, shape (n_classes,)
    number of training samples observed in each class.

class_prior_ : array, shape (n_classes,)
    probability of each class.

classes_ : array, shape (n_classes,)
    class labels known to the classifier

epsilon_ : float
    absolute additive value to variances

sigma_ : array, shape (n_classes, n_features)
    variance of each feature per class

theta_ : array, shape (n_classes, n_features)
    mean of each feature per class

Examples
--------
>>> import numpy as np
>>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
>>> Y = np.array([1, 1, 1, 2, 2, 2])
>>> from sklearn.naive_bayes import GaussianNB
>>> clf = GaussianNB()
>>> clf.fit(X, Y)
GaussianNB()
>>> print(clf.predict([[-0.8, -1]]))
[1]
>>> clf_pf = GaussianNB()
>>> clf_pf.partial_fit(X, Y, np.unique(Y))
GaussianNB()
>>> print(clf_pf.predict([[-0.8, -1]]))
[1]
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/naive_bayes.py#L184)
> `fit(self, X, y, sample_weight=None)`

Fit Gaussian Naive Bayes according to X, y

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training vectors, where n_samples is the number of samples
    and n_features is the number of features.

y : array-like, shape (n_samples,)
    Target values.

sample_weight : array-like, shape (n_samples,), optional (default=None)
    Weights applied to individual samples (1. for unweighted).

    .. versionadded:: 0.17
       Gaussian Naive Bayes supports fitting with *sample_weight*.

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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/naive_bayes.py#L287)
> `partial_fit(self, X, y, classes=None, sample_weight=None)`

Incremental fit on a batch of samples.

This method is expected to be called several times consecutively
on different chunks of a dataset so as to implement out-of-core
or online learning.

This is especially useful when the whole dataset is too big to fit in
memory at once.

This method has some performance and numerical stability overhead,
hence it is better to call partial_fit on chunks of data that are
as large as possible (as long as fitting in the memory budget) to
hide the overhead.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

y : array-like, shape (n_samples,)
    Target values.

classes : array-like, shape (n_classes,), optional (default=None)
    List of all the classes that can possibly appear in the y vector.

    Must be provided at the first call to partial_fit, can be omitted
    in subsequent calls.

sample_weight : array-like, shape (n_samples,), optional (default=None)
    Weights applied to individual samples (1. for unweighted).

    .. versionadded:: 0.17

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

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1195)
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

