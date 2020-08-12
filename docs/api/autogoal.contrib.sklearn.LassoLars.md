# `autogoal.contrib.sklearn.LassoLars`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L665)
> `LassoLars(self, fit_intercept, normalize, precompute, fit_path, positive)`

Lasso model fit with Least Angle Regression a.k.a. Lars

It is a Linear Model trained with an L1 prior as regularizer.

The optimization objective for Lasso is::

(1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

Read more in the :ref:`User Guide <least_angle_regression>`.

Parameters
----------
alpha : float, default=1.0
    Constant that multiplies the penalty term. Defaults to 1.0.
    ``alpha = 0`` is equivalent to an ordinary least square, solved
    by :class:`LinearRegression`. For numerical reasons, using
    ``alpha = 0`` with the LassoLars object is not advised and you
    should prefer the LinearRegression object.

fit_intercept : bool, default=True
    whether to calculate the intercept for this model. If set
    to false, no intercept will be used in calculations
    (i.e. data is expected to be centered).

verbose : bool or int, default=False
    Sets the verbosity amount

normalize : bool, default=True
    This parameter is ignored when ``fit_intercept`` is set to False.
    If True, the regressors X will be normalized before regression by
    subtracting the mean and dividing by the l2-norm.
    If you wish to standardize, please use
    :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
    on an estimator with ``normalize=False``.

precompute : bool, 'auto' or array-like, default='auto'
    Whether to use a precomputed Gram matrix to speed up
    calculations. If set to ``'auto'`` let us decide. The Gram
    matrix can also be passed as argument.

max_iter : int, default=500
    Maximum number of iterations to perform.

eps : float, optional
    The machine-precision regularization in the computation of the
    Cholesky diagonal factors. Increase this for very ill-conditioned
    systems. Unlike the ``tol`` parameter in some iterative
    optimization-based algorithms, this parameter does not control
    the tolerance of the optimization.
    By default, ``np.finfo(np.float).eps`` is used.

copy_X : bool, default=True
    If True, X will be copied; else, it may be overwritten.

fit_path : bool, default=True
    If ``True`` the full path is stored in the ``coef_path_`` attribute.
    If you compute the solution for a large problem or many targets,
    setting ``fit_path`` to ``False`` will lead to a speedup, especially
    with a small alpha.

positive : bool, default=False
    Restrict coefficients to be >= 0. Be aware that you might want to
    remove fit_intercept which is set True by default.
    Under the positive restriction the model coefficients will not converge
    to the ordinary-least-squares solution for small values of alpha.
    Only coefficients up to the smallest alpha value (``alphas_[alphas_ >
    0.].min()`` when fit_path=True) reached by the stepwise Lars-Lasso
    algorithm are typically in congruence with the solution of the
    coordinate descent Lasso estimator.

Attributes
----------
alphas_ : array-like of shape (n_alphas + 1,) | list of n_targets such             arrays
    Maximum of covariances (in absolute value) at each iteration.         ``n_alphas`` is either ``max_iter``, ``n_features``, or the number of         nodes in the path with correlation greater than ``alpha``, whichever         is smaller.

active_ : list, length = n_alphas | list of n_targets such lists
    Indices of active variables at the end of the path.

coef_path_ : array-like of shape (n_features, n_alphas + 1) or list
    If a list is passed it's expected to be one of n_targets such arrays.
    The varying values of the coefficients along the path. It is not
    present if the ``fit_path`` parameter is ``False``.

coef_ : array-like of shape (n_features,) or (n_targets, n_features)
    Parameter vector (w in the formulation formula).

intercept_ : float or array-like of shape (n_targets,)
    Independent term in decision function.

n_iter_ : array-like or int.
    The number of iterations taken by lars_path to find the
    grid of alphas for each target.

Examples
--------
>>> from sklearn import linear_model
>>> reg = linear_model.LassoLars(alpha=0.01)
>>> reg.fit([[-1, 1], [0, 0], [1, 1]], [-1, 0, -1])
LassoLars(alpha=0.01)
>>> print(reg.coef_)
[ 0.         -0.963257...]

See also
--------
lars_path
lasso_path
Lasso
LassoCV
LassoLarsCV
LassoLarsIC
sklearn.decomposition.sparse_encode
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_least_angle.py#L936)
> `fit(self, X, y, Xy=None)`

Fit the model using X, y as training data.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Training data.

y : array-like of shape (n_samples,) or (n_samples, n_targets)
    Target values.

Xy : array-like of shape (n_samples,) or (n_samples, n_targets),                 default=None
    Xy = np.dot(X.T, y) that can be precomputed. It is useful
    only when the Gram matrix is precomputed.

Returns
-------
self : object
    returns an instance of self.
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_base.py#L211)
> `predict(self, X)`

Predict using the linear model.

Parameters
----------
X : array_like or sparse matrix, shape (n_samples, n_features)
    Samples.

Returns
-------
C : array, shape (n_samples,)
    Returns predicted values.
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L684)
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

