# `autogoal.contrib.sklearn.ARDRegression`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L495)
> `ARDRegression(self, n_iter, tol, compute_score, threshold_lambda, fit_intercept, normalize)`

Bayesian ARD regression.

Fit the weights of a regression model, using an ARD prior. The weights of
the regression model are assumed to be in Gaussian distributions.
Also estimate the parameters lambda (precisions of the distributions of the
weights) and alpha (precision of the distribution of the noise).
The estimation is done by an iterative procedures (Evidence Maximization)

Read more in the :ref:`User Guide <bayesian_regression>`.

Parameters
----------
n_iter : int, default=300
    Maximum number of iterations.

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

compute_score : bool, default=False
    If True, compute the objective function at each step of the model.

threshold_lambda : float, default=10 000
    threshold for removing (pruning) weights with high precision from
    the computation.

fit_intercept : bool, default=True
    whether to calculate the intercept for this model. If set
    to false, no intercept will be used in calculations
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

alpha_ : float
   estimated precision of the noise.

lambda_ : array-like of shape (n_features,)
   estimated precisions of the weights.

sigma_ : array-like of shape (n_features, n_features)
    estimated variance-covariance matrix of the weights

scores_ : float
    if computed, value of the objective function (to be maximized)

intercept_ : float
    Independent term in decision function. Set to 0.0 if
    ``fit_intercept = False``.

Examples
--------
>>> from sklearn import linear_model
>>> clf = linear_model.ARDRegression()
>>> clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
ARDRegression()
>>> clf.predict([[1, 1]])
array([1.])

Notes
-----
For an example, see :ref:`examples/linear_model/plot_ard.py
<sphx_glr_auto_examples_linear_model_plot_ard.py>`.

References
----------
D. J. C. MacKay, Bayesian nonlinear modeling for the prediction
competition, ASHRAE Transactions, 1994.

R. Salakhutdinov, Lecture notes on Statistical Machine Learning,
http://www.utstat.toronto.edu/~rsalakhu/sta4273/notes/Lecture2.pdf#page=15
Their beta is our ``self.alpha_``
Their alpha is our ``self.lambda_``
ARD is a little different than the slide: only dimensions/features for
which ``self.lambda_ < self.threshold_lambda`` are kept and the rest are
discarded.
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_bayes.py#L511)
> `fit(self, X, y)`

Fit the ARDRegression model according to the given training data
and parameters.

Iterative procedure to maximize the evidence

Parameters
----------
X : array-like of shape (n_samples, n_features)
    Training vector, where n_samples in the number of samples and
    n_features is the number of features.
y : array-like of shape (n_samples,)
    Target values (integers). Will be cast to X's dtype if necessary

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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_bayes.py#L620)
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

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L516)
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

