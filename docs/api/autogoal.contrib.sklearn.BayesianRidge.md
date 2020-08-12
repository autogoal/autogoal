# `autogoal.contrib.sklearn.BayesianRidge`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L526)
> `BayesianRidge(self, n_iter, tol, compute_score, fit_intercept, normalize)`

Bayesian ridge regression.

Fit a Bayesian ridge model. See the Notes section for details on this
implementation and the optimization of the regularization parameters
lambda (precision of the weights) and alpha (precision of the noise).

Read more in the :ref:`User Guide <bayesian_regression>`.

Parameters
----------
n_iter : int, default=300
    Maximum number of iterations. Should be greater than or equal to 1.

tol : float, default=1e-3
    Stop the algorithm if w has converged.

alpha_1 : float, default=1e-6
    Hyper-parameter : shape parameter for the Gamma distribution prior
    over the alpha parameter.

alpha_2 : float, default=1e-6
    Hyper-parameter : inverse scale parameter (rate parameter) for the
    Gamma distribution prior over the alpha parameter.

lambda_1 : float, default=1e-6
    Hyper-parameter : shape parameter for the Gamma distribution prior
    over the lambda parameter.

lambda_2 : float, default=1e-6
    Hyper-parameter : inverse scale parameter (rate parameter) for the
    Gamma distribution prior over the lambda parameter.

alpha_init : float, default=None
    Initial value for alpha (precision of the noise).
    If not set, alpha_init is 1/Var(y).

        .. versionadded:: 0.22

lambda_init : float, default=None
    Initial value for lambda (precision of the weights).
    If not set, lambda_init is 1.

        .. versionadded:: 0.22

compute_score : bool, default=False
    If True, compute the log marginal likelihood at each iteration of the
    optimization.

fit_intercept : bool, default=True
    Whether to calculate the intercept for this model.
    The intercept is not treated as a probabilistic parameter
    and thus has no associated variance. If set
    to False, no intercept will be used in calculations
    (i.e. data is expected to be centered).

normalize : bool, default=False
    This parameter is ignored when ``fit_intercept`` is set to False.
    If True, the regressors X will be normalized before regression by
    subtracting the mean and dividing by the l2-norm.
    If you wish to standardize, please use
    :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
    on an estimator with ``normalize=False``.

copy_X : bool, default=True
    If True, X will be copied; else, it may be overwritten.

verbose : bool, default=False
    Verbose mode when fitting the model.


Attributes
----------
coef_ : array-like of shape (n_features,)
    Coefficients of the regression model (mean of distribution)

intercept_ : float
    Independent term in decision function. Set to 0.0 if
    ``fit_intercept = False``.

alpha_ : float
   Estimated precision of the noise.

lambda_ : float
   Estimated precision of the weights.

sigma_ : array-like of shape (n_features, n_features)
    Estimated variance-covariance matrix of the weights

scores_ : array-like of shape (n_iter_+1,)
    If computed_score is True, value of the log marginal likelihood (to be
    maximized) at each iteration of the optimization. The array starts
    with the value of the log marginal likelihood obtained for the initial
    values of alpha and lambda and ends with the value obtained for the
    estimated alpha and lambda.

n_iter_ : int
    The actual number of iterations to reach the stopping criterion.

Examples
--------
>>> from sklearn import linear_model
>>> clf = linear_model.BayesianRidge()
>>> clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
BayesianRidge()
>>> clf.predict([[1, 1]])
array([1.])

Notes
-----
There exist several strategies to perform Bayesian ridge regression. This
implementation is based on the algorithm described in Appendix A of
(Tipping, 2001) where updates of the regularization parameters are done as
suggested in (MacKay, 1992). Note that according to A New
View of Automatic Relevance Determination (Wipf and Nagarajan, 2008) these
update rules do not guarantee that the marginal likelihood is increasing
between two consecutive iterations of the optimization.

References
----------
D. J. C. MacKay, Bayesian Interpolation, Computation and Neural Systems,
Vol. 4, No. 3, 1992.

M. E. Tipping, Sparse Bayesian Learning and the Relevance Vector Machine,
Journal of Machine Learning Research, Vol. 1, 2001.
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_bayes.py#L168)
> `fit(self, X, y, sample_weight=None)`

Fit the model

Parameters
----------
X : ndarray of shape (n_samples, n_features)
    Training data
y : ndarray of shape (n_samples,)
    Target values. Will be cast to X's dtype if necessary

sample_weight : ndarray of shape (n_samples,), default=None
    Individual weights for each sample

    .. versionadded:: 0.20
       parameter *sample_weight* support to BayesianRidge.

Returns
-------
self : returns an instance of self.
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_bayes.py#L294)
> `predict(self, X, return_std=False)`

Predict using the linear model.

In addition to the mean of the predictive distribution, also its
standard deviation can be returned.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Samples.

return_std : bool, default=False
    Whether to return the standard deviation of posterior prediction.

Returns
-------
y_mean : array-like of shape (n_samples,)
    Mean of predictive distribution of query points.

y_std : array-like of shape (n_samples,)
    Standard deviation of predictive distribution of query points.
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L545)
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

