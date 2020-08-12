# `autogoal.contrib.sklearn.PassiveAggressiveRegressor`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L820)
> `PassiveAggressiveRegressor(self, C, fit_intercept, tol, early_stopping, validation_fraction, n_iter_no_change, shuffle, epsilon, average)`

Passive Aggressive Regressor

Read more in the :ref:`User Guide <passive_aggressive>`.

Parameters
----------

C : float
    Maximum step size (regularization). Defaults to 1.0.

fit_intercept : bool
    Whether the intercept should be estimated or not. If False, the
    data is assumed to be already centered. Defaults to True.

max_iter : int, optional (default=1000)
    The maximum number of passes over the training data (aka epochs).
    It only impacts the behavior in the ``fit`` method, and not the
    :meth:`partial_fit` method.

    .. versionadded:: 0.19

tol : float or None, optional (default=1e-3)
    The stopping criterion. If it is not None, the iterations will stop
    when (loss > previous_loss - tol).

    .. versionadded:: 0.19

early_stopping : bool, default=False
    Whether to use early stopping to terminate training when validation.
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

shuffle : bool, default=True
    Whether or not the training data should be shuffled after each epoch.

verbose : integer, optional
    The verbosity level

loss : string, optional
    The loss function to be used:
    epsilon_insensitive: equivalent to PA-I in the reference paper.
    squared_epsilon_insensitive: equivalent to PA-II in the reference
    paper.

epsilon : float
    If the difference between the current prediction and the correct label
    is below this threshold, the model is not updated.

random_state : int, RandomState instance or None, optional, default=None
    The seed of the pseudo random number generator to use when shuffling
    the data.  If int, random_state is the seed used by the random number
    generator; If RandomState instance, random_state is the random number
    generator; If None, the random number generator is the RandomState
    instance used by `np.random`.

warm_start : bool, optional
    When set to True, reuse the solution of the previous call to fit as
    initialization, otherwise, just erase the previous solution.
    See :term:`the Glossary <warm_start>`.

    Repeatedly calling fit or partial_fit when warm_start is True can
    result in a different solution than when calling fit a single time
    because of the way the data is shuffled.

average : bool or int, optional
    When set to True, computes the averaged SGD weights and stores the
    result in the ``coef_`` attribute. If set to an int greater than 1,
    averaging will begin once the total number of samples seen reaches
    average. So average=10 will begin averaging after seeing 10 samples.

    .. versionadded:: 0.19
       parameter *average* to use weights averaging in SGD

Attributes
----------
coef_ : array, shape = [1, n_features] if n_classes == 2 else [n_classes,            n_features]
    Weights assigned to the features.

intercept_ : array, shape = [1] if n_classes == 2 else [n_classes]
    Constants in decision function.

n_iter_ : int
    The actual number of iterations to reach the stopping criterion.

t_ : int
    Number of weight updates performed during training.
    Same as ``(n_iter_ * n_samples)``.

Examples
--------
>>> from sklearn.linear_model import PassiveAggressiveRegressor
>>> from sklearn.datasets import make_regression

>>> X, y = make_regression(n_features=4, random_state=0)
>>> regr = PassiveAggressiveRegressor(max_iter=100, random_state=0,
... tol=1e-3)
>>> regr.fit(X, y)
PassiveAggressiveRegressor(max_iter=100, random_state=0)
>>> print(regr.coef_)
[20.48736655 34.18818427 67.59122734 87.94731329]
>>> print(regr.intercept_)
[-0.02306214]
>>> print(regr.predict([[0, 0, 0, 0]]))
[-0.02306214]

See also
--------

SGDRegressor

References
----------
Online Passive-Aggressive Algorithms
<http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf>
K. Crammer, O. Dekel, J. Keshat, S. Shalev-Shwartz, Y. Singer - JMLR (2006)
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_passive_aggressive.py#L440)
> `fit(self, X, y, coef_init=None, intercept_init=None)`

Fit linear model with Passive Aggressive algorithm.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Training data

y : numpy array of shape [n_samples]
    Target values

coef_init : array, shape = [n_features]
    The initial coefficients to warm-start the optimization.

intercept_init : array, shape = [1]
    The initial intercept to warm-start the optimization.

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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/linear_model/_passive_aggressive.py#L417)
> `partial_fit(self, X, y)`

Fit linear model with Passive Aggressive algorithm.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Subset of training data

y : numpy array of shape [n_samples]
    Subset of target values

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

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L847)
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

