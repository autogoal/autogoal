# `autogoal.contrib.sklearn.NuSVR`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1647)
> `NuSVR(self, C, kernel, degree, gamma, coef0, shrinking, cache_size)`

Nu Support Vector Regression.

Similar to NuSVC, for regression, uses a parameter nu to control
the number of support vectors. However, unlike NuSVC, where nu
replaces C, here nu replaces the parameter epsilon of epsilon-SVR.

The implementation is based on libsvm.

Read more in the :ref:`User Guide <svm_regression>`.

Parameters
----------
nu : float, optional
    An upper bound on the fraction of training errors and a lower bound of
    the fraction of support vectors. Should be in the interval (0, 1].  By
    default 0.5 will be taken.

C : float, optional (default=1.0)
    Penalty parameter C of the error term.

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

tol : float, optional (default=1e-3)
    Tolerance for stopping criterion.

cache_size : float, optional
    Specify the size of the kernel cache (in MB).

verbose : bool, default: False
    Enable verbose output. Note that this setting takes advantage of a
    per-process runtime setting in libsvm that, if enabled, may not work
    properly in a multithreaded context.

max_iter : int, optional (default=-1)
    Hard limit on iterations within solver, or -1 for no limit.

Attributes
----------
support_ : array-like of shape (n_SV)
    Indices of support vectors.

support_vectors_ : array-like of shape (n_SV, n_features)
    Support vectors.

dual_coef_ : array, shape = [1, n_SV]
    Coefficients of the support vector in the decision function.

coef_ : array, shape = [1, n_features]
    Weights assigned to the features (coefficients in the primal
    problem). This is only available in the case of a linear kernel.

    `coef_` is readonly property derived from `dual_coef_` and
    `support_vectors_`.

intercept_ : array, shape = [1]
    Constants in decision function.

Examples
--------
>>> from sklearn.svm import NuSVR
>>> import numpy as np
>>> n_samples, n_features = 10, 5
>>> np.random.seed(0)
>>> y = np.random.randn(n_samples)
>>> X = np.random.randn(n_samples, n_features)
>>> clf = NuSVR(C=1.0, nu=0.1)
>>> clf.fit(X, y)
NuSVR(nu=0.1)

See also
--------
NuSVC
    Support Vector Machine for classification implemented with libsvm
    with a parameter to control the number of support vectors.

SVR
    epsilon Support Vector Machine for regression implemented with libsvm.

Notes
-----
**References:**
`LIBSVM: A Library for Support Vector Machines
<http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf>`__
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/svm/_base.py#L300)
> `predict(self, X)`

Perform regression on samples in X.

For an one-class model, +1 (inlier) or -1 (outlier) is returned.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    For kernel="precomputed", the expected shape of X is
    (n_samples_test, n_samples_train).

Returns
-------
y_pred : array, shape (n_samples,)
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1670)
> `run(self, input)`

### `score`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/base.py#L376)
> `score(self, X, y, sample_weight=None)`

Return the coefficient of determination R^2 of the prediction.

The coefficient R^2 is defined as (1 - u/v), where u is the residual
sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
sum of squares ((y_true - y_true.mean()) ** 2).sum().
The best possible score is 1.0 and it can be negative (because the
model can be arbitrarily worse). A constant model that always
predicts the expected value of y, disregarding the input features,
would get a R^2 score of 0.0.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Test samples. For some estimators this may be a
    precomputed kernel matrix or a list of generic objects instead,
    shape = (n_samples, n_samples_fitted),
    where n_samples_fitted is the number of
    samples used in the fitting for the estimator.

y : array-like of shape (n_samples,) or (n_samples, n_outputs)
    True values for X.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.

Returns
-------
score : float
    R^2 of self.predict(X) wrt. y.

Notes
-----
The R2 score used when calling ``score`` on a regressor will use
``multioutput='uniform_average'`` from version 0.23 to keep consistent
with :func:`~sklearn.metrics.r2_score`. This will influence the
``score`` method of all the multioutput regressors (except for
:class:`~sklearn.multioutput.MultiOutputRegressor`). To specify the
default value manually and avoid the warning, please either call
:func:`~sklearn.metrics.r2_score` directly or make a custom scorer with
:func:`~sklearn.metrics.make_scorer` (the built-in scorer ``'r2'`` uses
``multioutput='uniform_average'``).
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

