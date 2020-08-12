# `autogoal.contrib.sklearn.NuSVC`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1610)
> `NuSVC(self, kernel, degree, gamma, coef0, shrinking, probability, cache_size, decision_function_shape, break_ties)`

Nu-Support Vector Classification.

Similar to SVC but uses a parameter to control the number of support
vectors.

The implementation is based on libsvm.

Read more in the :ref:`User Guide <svm_classification>`.

Parameters
----------
nu : float, optional (default=0.5)
    An upper bound on the fraction of training errors and a lower
    bound of the fraction of support vectors. Should be in the
    interval (0, 1].

kernel : string, optional (default='rbf')
     Specifies the kernel type to be used in the algorithm.
     It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or
     a callable.
     If none is given, 'rbf' will be used. If a callable is given it is
     used to precompute the kernel matrix.

degree : int, optional (default=3)
    Degree of the polynomial kernel function ('poly').
    Ignored by all other kernels.

gamma : {'scale', 'auto'} or float, optional (default='scale')
    Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

    - if ``gamma='scale'`` (default) is passed then it uses
      1 / (n_features * X.var()) as value of gamma,
    - if 'auto', uses 1 / n_features.

    .. versionchanged:: 0.22
       The default value of ``gamma`` changed from 'auto' to 'scale'.

coef0 : float, optional (default=0.0)
    Independent term in kernel function.
    It is only significant in 'poly' and 'sigmoid'.

shrinking : boolean, optional (default=True)
    Whether to use the shrinking heuristic.

probability : boolean, optional (default=False)
    Whether to enable probability estimates. This must be enabled prior
    to calling `fit`, will slow down that method as it internally uses
    5-fold cross-validation, and `predict_proba` may be inconsistent with
    `predict`. Read more in the :ref:`User Guide <scores_probabilities>`.

tol : float, optional (default=1e-3)
    Tolerance for stopping criterion.

cache_size : float, optional
    Specify the size of the kernel cache (in MB).

class_weight : {dict, 'balanced'}, optional
    Set the parameter C of class i to class_weight[i]*C for
    SVC. If not given, all classes are supposed to have
    weight one. The "balanced" mode uses the values of y to automatically
    adjust weights inversely proportional to class frequencies as
    ``n_samples / (n_classes * np.bincount(y))``

verbose : bool, default: False
    Enable verbose output. Note that this setting takes advantage of a
    per-process runtime setting in libsvm that, if enabled, may not work
    properly in a multithreaded context.

max_iter : int, optional (default=-1)
    Hard limit on iterations within solver, or -1 for no limit.

decision_function_shape : 'ovo', 'ovr', default='ovr'
    Whether to return a one-vs-rest ('ovr') decision function of shape
    (n_samples, n_classes) as all other classifiers, or the original
    one-vs-one ('ovo') decision function of libsvm which has shape
    (n_samples, n_classes * (n_classes - 1) / 2).

    .. versionchanged:: 0.19
        decision_function_shape is 'ovr' by default.

    .. versionadded:: 0.17
       *decision_function_shape='ovr'* is recommended.

    .. versionchanged:: 0.17
       Deprecated *decision_function_shape='ovo' and None*.

break_ties : bool, optional (default=False)
    If true, ``decision_function_shape='ovr'``, and number of classes > 2,
    :term:`predict` will break ties according to the confidence values of
    :term:`decision_function`; otherwise the first class among the tied
    classes is returned. Please note that breaking ties comes at a
    relatively high computational cost compared to a simple predict.

    .. versionadded:: 0.22

random_state : int, RandomState instance or None, optional (default=None)
    The seed of the pseudo random number generator used when shuffling
    the data for probability estimates. If int, random_state is the seed
    used by the random number generator; If RandomState instance,
    random_state is the random number generator; If None, the random
    number generator is the RandomState instance used by `np.random`.

Attributes
----------
support_ : array-like of shape (n_SV)
    Indices of support vectors.

support_vectors_ : array-like of shape (n_SV, n_features)
    Support vectors.

n_support_ : array-like, dtype=int32, shape = [n_class]
    Number of support vectors for each class.

dual_coef_ : array, shape = [n_class-1, n_SV]
    Coefficients of the support vector in the decision function.
    For multiclass, coefficient for all 1-vs-1 classifiers.
    The layout of the coefficients in the multiclass case is somewhat
    non-trivial. See the section about multi-class classification in
    the SVM section of the User Guide for details.

coef_ : array, shape = [n_class * (n_class-1) / 2, n_features]
    Weights assigned to the features (coefficients in the primal
    problem). This is only available in the case of a linear kernel.

    `coef_` is readonly property derived from `dual_coef_` and
    `support_vectors_`.

intercept_ : ndarray of shape (n_class * (n_class-1) / 2,)
    Constants in decision function.

classes_ : array of shape (n_classes,)
    The unique classes labels.

fit_status_ : int
    0 if correctly fitted, 1 if the algorithm did not converge.

probA_ : ndarray, shape of (n_class * (n_class-1) / 2,)
probB_ : ndarray of shape (n_class * (n_class-1) / 2,)
    If `probability=True`, it corresponds to the parameters learned in
    Platt scaling to produce probability estimates from decision values.
    If `probability=False`, it's an empty array. Platt scaling uses the
    logistic function
    ``1 / (1 + exp(decision_value * probA_ + probB_))``
    where ``probA_`` and ``probB_`` are learned from the dataset [2]_. For
    more information on the multiclass case and training procedure see
    section 8 of [1]_.

class_weight_ : ndarray of shape (n_class,)
    Multipliers of parameter C of each class.
    Computed based on the ``class_weight`` parameter.

shape_fit_ : tuple of int of shape (n_dimensions_of_X,)
    Array dimensions of training vector ``X``.

Examples
--------
>>> import numpy as np
>>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
>>> y = np.array([1, 1, 2, 2])
>>> from sklearn.svm import NuSVC
>>> clf = NuSVC()
>>> clf.fit(X, y)
NuSVC()
>>> print(clf.predict([[-0.8, -1]]))
[1]

See also
--------
SVC
    Support Vector Machine for classification using libsvm.

LinearSVC
    Scalable linear Support Vector Machine for classification using
    liblinear.

References
----------
.. [1] `LIBSVM: A Library for Support Vector Machines
    <http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf>`_

.. [2] `Platt, John (1999). "Probabilistic outputs for support vector
    machines and comparison to regularizedlikelihood methods."
    <http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.41.1639>`_
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `decision_function`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/svm/_base.py#L537)
> `decision_function(self, X)`

Evaluates the decision function for the samples in X.

Parameters
----------
X : array-like, shape (n_samples, n_features)

Returns
-------
X : array-like, shape (n_samples, n_classes * (n_classes-1) / 2)
    Returns the decision function of the sample for each class
    in the model.
    If decision_function_shape='ovr', the shape is (n_samples,
    n_classes).

Notes
-----
If decision_function_shape='ovo', the function values are proportional
to the distance of the samples X to the separating hyperplane. If the
exact distances are required, divide the function values by the norm of
the weight vector (``coef_``). See also `this question
<https://stats.stackexchange.com/questions/14876/
interpreting-distance-from-hyperplane-in-svm>`_ for further details.
If decision_function_shape='ovr', the decision function is a monotonic
transformation of ovo decision function.
### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/svm/_base.py#L107)
> `fit(self, X, y, sample_weight=None)`

Fit the SVM model according to the given training data.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Training vectors, where n_samples is the number of samples
    and n_features is the number of features.
    For kernel="precomputed", the expected shape of X is
    (n_samples, n_samples).

y : array-like, shape (n_samples,)
    Target values (class labels in classification, real numbers in
    regression)

sample_weight : array-like, shape (n_samples,)
    Per-sample weights. Rescale C per sample. Higher weights
    force the classifier to put more emphasis on these points.

Returns
-------
self : object

Notes
-----
If X and y are not C-ordered and contiguous arrays of np.float64 and
X is not a scipy.sparse.csr_matrix, X and/or y may be copied.

If X is a dense array, then the other methods will not support sparse
matrices as input.
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/svm/_base.py#L568)
> `predict(self, X)`

Perform classification on samples in X.

For an one-class model, +1 or -1 is returned.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    For kernel="precomputed", the expected shape of X is
    [n_samples_test, n_samples_train]

Returns
-------
y_pred : array, shape (n_samples,)
    Class labels for samples in X.
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1637)
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

