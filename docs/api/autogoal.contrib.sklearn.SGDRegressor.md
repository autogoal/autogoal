# `autogoal.contrib.sklearn.SGDRegressor`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1009)
> `SGDRegressor(self, loss, penalty, l1_ratio, fit_intercept, tol, shuffle, epsilon, learning_rate, eta0, power_t, early_stopping, validation_fraction, n_iter_no_change, average)`

Linear model fitted by minimizing a regularized empirical loss with SGD

SGD stands for Stochastic Gradient Descent: the gradient of the loss is
estimated each sample at a time and the model is updated along the way with
a decreasing strength schedule (aka learning rate).

The regularizer is a penalty added to the loss function that shrinks model
parameters towards the zero vector using either the squared euclidean norm
L2 or the absolute norm L1 or a combination of both (Elastic Net). If the
parameter update crosses the 0.0 value because of the regularizer, the
update is truncated to 0.0 to allow for learning sparse models and achieve
online feature selection.

This implementation works with data represented as dense numpy arrays of
floating point values for the features.

Read more in the :ref:`User Guide <sgd>`.

Parameters
----------
loss : str, default='squared_loss'
    The loss function to be used. The possible values are 'squared_loss',
    'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'

    The 'squared_loss' refers to the ordinary least squares fit.
    'huber' modifies 'squared_loss' to focus less on getting outliers
    correct by switching from squared to linear loss past a distance of
    epsilon. 'epsilon_insensitive' ignores errors less than epsilon and is
    linear past that; this is the loss function used in SVR.
    'squared_epsilon_insensitive' is the same but becomes squared loss past
    a tolerance of epsilon.

penalty : {'l2', 'l1', 'elasticnet'}, default='l2'
    The penalty (aka regularization term) to be used. Defaults to 'l2'
    which is the standard regularizer for linear SVM models. 'l1' and
    'elasticnet' might bring sparsity to the model (feature selection)
    not achievable with 'l2'.

alpha : float, default=0.0001
    Constant that multiplies the regularization term.
    Also used to compute learning_rate when set to 'optimal'.

l1_ratio : float, default=0.15
    The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1.
    l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.

fit_intercept : bool, default=True
    Whether the intercept should be estimated or not. If False, the
    data is assumed to be already centered.

max_iter : int, default=1000
    The maximum number of passes over the training data (aka epochs).
    It only impacts the behavior in the ``fit`` method, and not the
    :meth:`partial_fit` method.

    .. versionadded:: 0.19

tol : float, default=1e-3
    The stopping criterion. If it is not None, the iterations will stop
    when (loss > best_loss - tol) for ``n_iter_no_change`` consecutive
    epochs.

    .. versionadded:: 0.19

shuffle : bool, default=True
    Whether or not the training data should be shuffled after each epoch.

verbose : int, default=0
    The verbosity level.

epsilon : float, default=0.1
    Epsilon in the epsilon-insensitive loss functions; only if `loss` is
    'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'.
    For 'huber', determines the threshold at which it becomes less
    important to get the prediction exactly right.
    For epsilon-insensitive, any differences between the current prediction
    and the correct label are ignored if they are less than this threshold.

random_state : int, RandomState instance, default=None
    The seed of the pseudo random number generator to use when shuffling
    the data.  If int, random_state is the seed used by the random number
    generator; If RandomState instance, random_state is the random number
    generator; If None, the random number generator is the RandomState
    instance used by `np.random`.

learning_rate : string, default='invscaling'
    The learning rate schedule:

    'constant':
        eta = eta0
    'optimal':
        eta = 1.0 / (alpha * (t + t0))
        where t0 is chosen by a heuristic proposed by Leon Bottou.
    'invscaling': [default]
        eta = eta0 / pow(t, power_t)
    'adaptive':
        eta = eta0, as long as the training keeps decreasing.
        Each time n_iter_no_change consecutive epochs fail to decrease the
        training loss by tol or fail to increase validation score by tol if
        early_stopping is True, the current learning rate is divided by 5.

eta0 : double, default=0.01
    The initial learning rate for the 'constant', 'invscaling' or
    'adaptive' schedules. The default value is 0.01.

power_t : double, default=0.25
    The exponent for inverse scaling learning rate.

early_stopping : bool, default=False
    Whether to use early stopping to terminate training when validation
    score is not improving. If set to True, it will automatically set aside
    a fraction of training data as validation and terminate
    training when validation score is not improving by at least tol for
    n_iter_no_change consecutive epochs.

    .. versionadded:: 0.20

validation_fraction : float, default=0.1
    The proportion of training data to set aside as validation set for
    early stopping. Must be between 0 and 1.
    Only used if early_stopping is True.

    .. versionadded:: 0.20

n_iter_no_change : int, default=5
    Number of iterations with no improvement to wait before early stopping.

    .. versionadded:: 0.20

warm_start : bool, default=False
    When set to True, reuse the solution of the previous call to fit as
    initialization, otherwise, just erase the previous solution.
    See :term:`the Glossary <warm_start>`.

    Repeatedly calling fit or partial_fit when warm_start is True can
    result in a different solution than when calling fit a single time
    because of the way the data is shuffled.
    If a dynamic learning rate is used, the learning rate is adapted
    depending on the number of samples already seen. Calling ``fit`` resets
    this counter, while ``partial_fit``  will result in increasing the
    existing counter.

average : bool or int, default=False
    When set to True, computes the averaged SGD weights and stores the
    result in the ``coef_`` attribute. If set to an int greater than 1,
    averaging will begin once the total number of samples seen reaches
    average. So ``average=10`` will begin averaging after seeing 10
    samples.

Attributes
----------
coef_ : ndarray of shape (n_features,)
    Weights assigned to the features.

intercept_ : ndarray of shape (1,)
    The intercept term.

average_coef_ : ndarray of shape (n_features,)
    Averaged weights assigned to the features.

average_intercept_ : ndarray of shape (1,)
    The averaged intercept term.

n_iter_ : int
    The actual number of iterations to reach the stopping criterion.

t_ : int
    Number of weight updates performed during training.
    Same as ``(n_iter_ * n_samples)``.

Examples
--------
>>> import numpy as np
>>> from sklearn import linear_model
>>> n_samples, n_features = 10, 5
>>> rng = np.random.RandomState(0)
>>> y = rng.randn(n_samples)
>>> X = rng.randn(n_samples, n_features)
>>> clf = linear_model.SGDRegressor(max_iter=1000, tol=1e-3)
>>> clf.fit(X, y)
SGDRegressor()

See also
--------
Ridge, ElasticNet, Lasso, sklearn.svm.SVR
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `densify`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_base.py#L323)
> `densify(self)`

Convert coefficient matrix to dense array format.

Converts the ``coef_`` member (back) to a numpy.ndarray. This is the
default format of ``coef_`` and is required for fitting, so calling
this method is only required on models that have previously been
sparsified; otherwise, it is a no-op.

Returns
-------
self
    Fitted estimator.
### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_stochastic_gradient.py#L1191)
> `fit(self, X, y, coef_init=None, intercept_init=None, sample_weight=None)`

Fit linear model with Stochastic Gradient Descent.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Training data

y : ndarray of shape (n_samples,)
    Target values

coef_init : ndarray of shape (n_features,), default=None
    The initial coefficients to warm-start the optimization.

intercept_init : ndarray of shape (1,), default=None
    The initial intercept to warm-start the optimization.

sample_weight : array-like, shape (n_samples,), default=None
    Weights applied to individual samples (1. for unweighted).

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
### `partial_fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_stochastic_gradient.py#L1126)
> `partial_fit(self, X, y, sample_weight=None)`

Perform one epoch of stochastic gradient descent on given samples.

Internally, this method uses ``max_iter = 1``. Therefore, it is not
guaranteed that a minimum of the cost function is reached after calling
it once. Matters such as objective convergence and early stopping
should be handled by the user.

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)
    Subset of training data

y : numpy array of shape (n_samples,)
    Subset of target values

sample_weight : array-like, shape (n_samples,), default=None
    Weights applied to individual samples.
    If not provided, uniform weights are assumed.

Returns
-------
self : returns an instance of self.
### `predict`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_stochastic_gradient.py#L1242)
> `predict(self, X)`

Predict using the linear model

Parameters
----------
X : {array-like, sparse matrix}, shape (n_samples, n_features)

Returns
-------
ndarray of shape (n_samples,)
   Predicted target values per element in X.
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1051)
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_stochastic_gradient.py#L101)
> `set_params(self, **kwargs)`

Set and validate the parameters of estimator.

Parameters
----------
**kwargs : dict
    Estimator parameters.

Returns
-------
self : object
    Estimator instance.
### `sparsify`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_base.py#L343)
> `sparsify(self)`

Convert coefficient matrix to sparse format.

Converts the ``coef_`` member to a scipy.sparse matrix, which for
L1-regularized models can be much more memory- and storage-efficient
than the usual numpy.ndarray representation.

The ``intercept_`` member is not converted.

Returns
-------
self
    Fitted estimator.

Notes
-----
For non-sparse models, i.e. when there are not many zeros in ``coef_``,
this may actually *increase* memory usage, so use this method with
care. A rule of thumb is that the number of zero elements, which can
be computed with ``(coef_ == 0).sum()``, must be more than 50% for this
to provide significant benefits.

After calling this method, further fitting with the partial_fit
method (if any) will not work until you call densify.
### `train`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L47)
> `train(self)`

