# `autogoal.contrib.sklearn.LassoLarsIC`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L694)
> `LassoLarsIC(self, criterion, fit_intercept, normalize, precompute, positive)`

Lasso model fit with Lars using BIC or AIC for model selection

The optimization objective for Lasso is::

(1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

AIC is the Akaike information criterion and BIC is the Bayes
Information criterion. Such criteria are useful to select the value
of the regularization parameter by making a trade-off between the
goodness of fit and the complexity of the model. A good model should
explain well the data while being simple.

Read more in the :ref:`User Guide <least_angle_regression>`.

Parameters
----------
criterion : {'bic' , 'aic'}, default='aic'
    The type of criterion to use.

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
    Maximum number of iterations to perform. Can be used for
    early stopping.

eps : float, optional
    The machine-precision regularization in the computation of the
    Cholesky diagonal factors. Increase this for very ill-conditioned
    systems. Unlike the ``tol`` parameter in some iterative
    optimization-based algorithms, this parameter does not control
    the tolerance of the optimization.
    By default, ``np.finfo(np.float).eps`` is used

copy_X : bool, default=True
    If True, X will be copied; else, it may be overwritten.

positive : bool, default=False
    Restrict coefficients to be >= 0. Be aware that you might want to
    remove fit_intercept which is set True by default.
    Under the positive restriction the model coefficients do not converge
    to the ordinary-least-squares solution for small values of alpha.
    Only coefficients up to the smallest alpha value (``alphas_[alphas_ >
    0.].min()`` when fit_path=True) reached by the stepwise Lars-Lasso
    algorithm are typically in congruence with the solution of the
    coordinate descent Lasso estimator.
    As a consequence using LassoLarsIC only makes sense for problems where
    a sparse solution is expected and/or reached.

Attributes
----------
coef_ : array-like of shape (n_features,)
    parameter vector (w in the formulation formula)

intercept_ : float
    independent term in decision function.

alpha_ : float
    the alpha parameter chosen by the information criterion

n_iter_ : int
    number of iterations run by lars_path to find the grid of
    alphas.

criterion_ : array-like of shape (n_alphas,)
    The value of the information criteria ('aic', 'bic') across all
    alphas. The alpha which has the smallest information criterion is
    chosen. This value is larger by a factor of ``n_samples`` compared to
    Eqns. 2.15 and 2.16 in (Zou et al, 2007).


Examples
--------
>>> from sklearn import linear_model
>>> reg = linear_model.LassoLarsIC(criterion='bic')
>>> reg.fit([[-1, 1], [0, 0], [1, 1]], [-1.1111, 0, -1.1111])
LassoLarsIC(criterion='bic')
>>> print(reg.coef_)
[ 0.  -1.11...]

Notes
-----
The estimation of the number of degrees of freedom is given by:

"On the degrees of freedom of the lasso"
Hui Zou, Trevor Hastie, and Robert Tibshirani
Ann. Statist. Volume 35, Number 5 (2007), 2173-2192.

https://en.wikipedia.org/wiki/Akaike_information_criterion
https://en.wikipedia.org/wiki/Bayesian_information_criterion

See also
--------
lars_path, LassoLars, LassoLarsCV
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_least_angle.py#L1738)
> `fit(self, X, y, copy_X=None)`

Fit the model using X, y as training data.

Parameters
----------
X : array-like of shape (n_samples, n_features)
    training data.

y : array-like of shape (n_samples,)
    target values. Will be cast to X's dtype if necessary

copy_X : bool, default=None
    If provided, this parameter will override the choice
    of copy_X made at instance creation.
    If ``True``, X will be copied; else, it may be overwritten.

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

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L713)
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

