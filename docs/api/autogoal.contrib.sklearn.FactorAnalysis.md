# `autogoal.contrib.sklearn.FactorAnalysis`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L161)
> `FactorAnalysis(self, tol, svd_method, iterated_power)`

Factor Analysis (FA)

A simple linear generative model with Gaussian latent variables.

The observations are assumed to be caused by a linear transformation of
lower dimensional latent factors and added Gaussian noise.
Without loss of generality the factors are distributed according to a
Gaussian with zero mean and unit covariance. The noise is also zero mean
and has an arbitrary diagonal covariance matrix.

If we would restrict the model further, by assuming that the Gaussian
noise is even isotropic (all diagonal entries are the same) we would obtain
:class:`PPCA`.

FactorAnalysis performs a maximum likelihood estimate of the so-called
`loading` matrix, the transformation of the latent variables to the
observed ones, using SVD based approach.

Read more in the :ref:`User Guide <FA>`.

.. versionadded:: 0.13

Parameters
----------
n_components : int | None
    Dimensionality of latent space, the number of components
    of ``X`` that are obtained after ``transform``.
    If None, n_components is set to the number of features.

tol : float
    Stopping tolerance for log-likelihood increase.

copy : bool
    Whether to make a copy of X. If ``False``, the input X gets overwritten
    during fitting.

max_iter : int
    Maximum number of iterations.

noise_variance_init : None | array, shape=(n_features,)
    The initial guess of the noise variance for each feature.
    If None, it defaults to np.ones(n_features)

svd_method : {'lapack', 'randomized'}
    Which SVD method to use. If 'lapack' use standard SVD from
    scipy.linalg, if 'randomized' use fast ``randomized_svd`` function.
    Defaults to 'randomized'. For most applications 'randomized' will
    be sufficiently precise while providing significant speed gains.
    Accuracy can also be improved by setting higher values for
    `iterated_power`. If this is not sufficient, for maximum precision
    you should choose 'lapack'.

iterated_power : int, optional
    Number of iterations for the power method. 3 by default. Only used
    if ``svd_method`` equals 'randomized'

random_state : int, RandomState instance or None, optional (default=0)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`. Only used when ``svd_method`` equals 'randomized'.

Attributes
----------
components_ : array, [n_components, n_features]
    Components with maximum variance.

loglike_ : list, [n_iterations]
    The log likelihood at each iteration.

noise_variance_ : array, shape=(n_features,)
    The estimated noise variance for each feature.

n_iter_ : int
    Number of iterations run.

mean_ : array, shape (n_features,)
    Per-feature empirical mean, estimated from the training set.

Examples
--------
>>> from sklearn.datasets import load_digits
>>> from sklearn.decomposition import FactorAnalysis
>>> X, _ = load_digits(return_X_y=True)
>>> transformer = FactorAnalysis(n_components=7, random_state=0)
>>> X_transformed = transformer.fit_transform(X)
>>> X_transformed.shape
(1797, 7)

References
----------
.. David Barber, Bayesian Reasoning and Machine Learning,
    Algorithm 21.1

.. Christopher M. Bishop: Pattern Recognition and Machine Learning,
    Chapter 12.2.4

See also
--------
PCA: Principal component analysis is also a latent linear variable model
    which however assumes equal noise variance for each feature.
    This extra assumption makes probabilistic PCA faster as it can be
    computed in closed form.
FastICA: Independent component analysis, a latent variable model with
    non-Gaussian latent variables.
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/decomposition/_factor_analysis.py#L158)
> `fit(self, X, y=None)`

Fit the FactorAnalysis model to X using SVD based approach

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training data.

y : Ignored

Returns
-------
self
### `fit_transform`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/base.py#L544)
> `fit_transform(self, X, y=None, **fit_params)`

Fit to data, then transform it.

Fits transformer to X and y with optional parameters fit_params
and returns a transformed version of X.

Parameters
----------
X : numpy array of shape [n_samples, n_features]
    Training set.

y : numpy array of shape [n_samples]
    Target values.

**fit_params : dict
    Additional fit parameters.

Returns
-------
X_new : numpy array of shape [n_samples, n_features_new]
    Transformed array.
### `get_covariance`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/decomposition/_factor_analysis.py#L280)
> `get_covariance(self)`

Compute data covariance with the FactorAnalysis model.

``cov = components_.T * components_ + diag(noise_variance)``

Returns
-------
cov : array, shape (n_features, n_features)
    Estimated covariance of data.
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
### `get_precision`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/decomposition/_factor_analysis.py#L296)
> `get_precision(self)`

Compute data precision matrix with the FactorAnalysis model.

Returns
-------
precision : array, shape (n_features, n_features)
    Estimated precision of data.
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L173)
> `run(self, input)`

### `score`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/decomposition/_factor_analysis.py#L348)
> `score(self, X, y=None)`

Compute the average log-likelihood of the samples

Parameters
----------
X : array, shape (n_samples, n_features)
    The data

y : Ignored

Returns
-------
ll : float
    Average log-likelihood of the samples under the current model
### `score_samples`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/decomposition/_factor_analysis.py#L325)
> `score_samples(self, X)`

Compute the log-likelihood of each sample

Parameters
----------
X : array, shape (n_samples, n_features)
    The data

Returns
-------
ll : array, shape (n_samples,)
    Log-likelihood of each sample under the current model
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

### `transform`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/decomposition/_factor_analysis.py#L250)
> `transform(self, X)`

Apply dimensionality reduction to X using the model.

Compute the expected mean of the latent variables.
See Barber, 21.2.33 (or Bishop, 12.66).

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training data.

Returns
-------
X_new : array-like, shape (n_samples, n_components)
    The latent variables of X.
