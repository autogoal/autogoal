# `autogoal.contrib.sklearn.Lasso`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L588)
> `Lasso(self, alpha, fit_intercept, normalize, precompute, positive, selection)`

Linear Model trained with L1 prior as regularizer (aka the Lasso)

The optimization objective for Lasso is::

    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

Technically the Lasso model is optimizing the same objective function as
the Elastic Net with ``l1_ratio=1.0`` (no L2 penalty).

Read more in the :ref:`User Guide <lasso>`.

Parameters
----------
alpha : float, optional
    Constant that multiplies the L1 term. Defaults to 1.0.
    ``alpha = 0`` is equivalent to an ordinary least square, solved
    by the :class:`LinearRegression` object. For numerical
    reasons, using ``alpha = 0`` with the ``Lasso`` object is not advised.
    Given this, you should use the :class:`LinearRegression` object.

fit_intercept : boolean, optional, default True
    Whether to calculate the intercept for this model. If set
    to False, no intercept will be used in calculations
    (i.e. data is expected to be centered).

normalize : boolean, optional, default False
    This parameter is ignored when ``fit_intercept`` is set to False.
    If True, the regressors X will be normalized before regression by
    subtracting the mean and dividing by the l2-norm.
    If you wish to standardize, please use
    :class:`sklearn.preprocessing.StandardScaler` before calling ``fit``
    on an estimator with ``normalize=False``.

precompute : True | False | array-like, default=False
    Whether to use a precomputed Gram matrix to speed up
    calculations. If set to ``'auto'`` let us decide. The Gram
    matrix can also be passed as argument. For sparse input
    this option is always ``True`` to preserve sparsity.

copy_X : boolean, optional, default True
    If ``True``, X will be copied; else, it may be overwritten.

max_iter : int, optional
    The maximum number of iterations

tol : float, optional
    The tolerance for the optimization: if the updates are
    smaller than ``tol``, the optimization code checks the
    dual gap for optimality and continues until it is smaller
    than ``tol``.

warm_start : bool, optional
    When set to True, reuse the solution of the previous call to fit as
    initialization, otherwise, just erase the previous solution.
    See :term:`the Glossary <warm_start>`.

positive : bool, optional
    When set to ``True``, forces the coefficients to be positive.

random_state : int, RandomState instance or None, optional, default None
    The seed of the pseudo random number generator that selects a random
    feature to update.  If int, random_state is the seed used by the random
    number generator; If RandomState instance, random_state is the random
    number generator; If None, the random number generator is the
    RandomState instance used by `np.random`. Used when ``selection`` ==
    'random'.

selection : str, default 'cyclic'
    If set to 'random', a random coefficient is updated every iteration
    rather than looping over features sequentially by default. This
    (setting to 'random') often leads to significantly faster convergence
    especially when tol is higher than 1e-4.

Attributes
----------
coef_ : array, shape (n_features,) | (n_targets, n_features)
    parameter vector (w in the cost function formula)

sparse_coef_ : scipy.sparse matrix, shape (n_features, 1) |             (n_targets, n_features)
    ``sparse_coef_`` is a readonly property derived from ``coef_``

intercept_ : float | array, shape (n_targets,)
    independent term in decision function.

n_iter_ : int | array-like, shape (n_targets,)
    number of iterations run by the coordinate descent solver to reach
    the specified tolerance.

Examples
--------
>>> from sklearn import linear_model
>>> clf = linear_model.Lasso(alpha=0.1)
>>> clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
Lasso(alpha=0.1)
>>> print(clf.coef_)
[0.85 0.  ]
>>> print(clf.intercept_)
0.15...

See also
--------
lars_path
lasso_path
LassoLars
LassoCV
LassoLarsCV
sklearn.decomposition.sparse_encode

Notes
-----
The algorithm used to fit the model is coordinate descent.

To avoid unnecessary memory duplication the X argument of the fit method
should be directly passed as a Fortran-contiguous numpy array.
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_coordinate_descent.py#L659)
> `fit(self, X, y, check_input=True)`

Fit model with coordinate descent.

Parameters
----------
X : ndarray or scipy.sparse matrix, (n_samples, n_features)
    Data

y : ndarray, shape (n_samples,) or (n_samples, n_targets)
    Target. Will be cast to X's dtype if necessary

check_input : boolean, (default=True)
    Allow to bypass several input checking.
    Don't use this parameter unless you know what you do.

Notes
-----

Coordinate descent is an algorithm that considers each column of
data at a time hence it will automatically convert the X input
as a Fortran-contiguous numpy array if necessary.

To avoid memory re-allocation it is advised to allocate the
initial data in memory directly using that format.
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
### `enet_path`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_coordinate_descent.py#L266)
> `enet_path(X, y, l1_ratio=0.5, eps=0.001, n_alphas=100, alphas=None, precompute='auto', Xy=None, copy_X=True, coef_init=None, verbose=False, return_n_iter=False, positive=False, check_input=True, **params)`

Compute elastic net path with coordinate descent.

The elastic net optimization function varies for mono and multi-outputs.

For mono-output tasks it is::

    1 / (2 * n_samples) * ||y - Xw||^2_2
    + alpha * l1_ratio * ||w||_1
    + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2

For multi-output tasks it is::

    (1 / (2 * n_samples)) * ||Y - XW||^Fro_2
    + alpha * l1_ratio * ||W||_21
    + 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2

Where::

    ||W||_21 = \sum_i \sqrt{\sum_j w_{ij}^2}

i.e. the sum of norm of each row.

Read more in the :ref:`User Guide <elastic_net>`.

Parameters
----------
X : {array-like}, shape (n_samples, n_features)
    Training data. Pass directly as Fortran-contiguous data to avoid
    unnecessary memory duplication. If ``y`` is mono-output then ``X``
    can be sparse.

y : ndarray, shape (n_samples,) or (n_samples, n_outputs)
    Target values.

l1_ratio : float, optional
    Number between 0 and 1 passed to elastic net (scaling between
    l1 and l2 penalties). ``l1_ratio=1`` corresponds to the Lasso.

eps : float
    Length of the path. ``eps=1e-3`` means that
    ``alpha_min / alpha_max = 1e-3``.

n_alphas : int, optional
    Number of alphas along the regularization path.

alphas : ndarray, optional
    List of alphas where to compute the models.
    If None alphas are set automatically.

precompute : True | False | 'auto' | array-like
    Whether to use a precomputed Gram matrix to speed up
    calculations. If set to ``'auto'`` let us decide. The Gram
    matrix can also be passed as argument.

Xy : array-like, optional
    Xy = np.dot(X.T, y) that can be precomputed. It is useful
    only when the Gram matrix is precomputed.

copy_X : bool, optional, default True
    If ``True``, X will be copied; else, it may be overwritten.

coef_init : array, shape (n_features, ) | None
    The initial values of the coefficients.

verbose : bool or int
    Amount of verbosity.

return_n_iter : bool
    Whether to return the number of iterations or not.

positive : bool, default False
    If set to True, forces coefficients to be positive.
    (Only allowed when ``y.ndim == 1``).

check_input : bool, default True
    Skip input validation checks, including the Gram matrix when provided
    assuming there are handled by the caller when check_input=False.

**params : kwargs
    Keyword arguments passed to the coordinate descent solver.

Returns
-------
alphas : array, shape (n_alphas,)
    The alphas along the path where models are computed.

coefs : array, shape (n_features, n_alphas) or             (n_outputs, n_features, n_alphas)
    Coefficients along the path.

dual_gaps : array, shape (n_alphas,)
    The dual gaps at the end of the optimization for each alpha.

n_iters : array-like, shape (n_alphas,)
    The number of iterations taken by the coordinate descent optimizer to
    reach the specified tolerance for each alpha.
    (Is returned when ``return_n_iter`` is set to True).

See Also
--------
MultiTaskElasticNet
MultiTaskElasticNetCV
ElasticNet
ElasticNetCV

Notes
-----
For an example, see
:ref:`examples/linear_model/plot_lasso_coordinate_descent_path.py
<sphx_glr_auto_examples_linear_model_plot_lasso_coordinate_descent_path.py>`.
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

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L609)
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

