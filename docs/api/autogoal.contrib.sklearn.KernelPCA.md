# `autogoal.contrib.sklearn.KernelPCA`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L212)
> `KernelPCA(self, degree, alpha, fit_inverse_transform, eigen_solver, tol, remove_zero_eig)`

Kernel Principal component analysis (KPCA)

Non-linear dimensionality reduction through the use of kernels (see
:ref:`metrics`).

Read more in the :ref:`User Guide <kernel_PCA>`.

Parameters
----------
n_components : int, default=None
    Number of components. If None, all non-zero components are kept.

kernel : "linear" | "poly" | "rbf" | "sigmoid" | "cosine" | "precomputed"
    Kernel. Default="linear".

gamma : float, default=1/n_features
    Kernel coefficient for rbf, poly and sigmoid kernels. Ignored by other
    kernels.

degree : int, default=3
    Degree for poly kernels. Ignored by other kernels.

coef0 : float, default=1
    Independent term in poly and sigmoid kernels.
    Ignored by other kernels.

kernel_params : mapping of string to any, default=None
    Parameters (keyword arguments) and values for kernel passed as
    callable object. Ignored by other kernels.

alpha : int, default=1.0
    Hyperparameter of the ridge regression that learns the
    inverse transform (when fit_inverse_transform=True).

fit_inverse_transform : bool, default=False
    Learn the inverse transform for non-precomputed kernels.
    (i.e. learn to find the pre-image of a point)

eigen_solver : string ['auto'|'dense'|'arpack'], default='auto'
    Select eigensolver to use. If n_components is much less than
    the number of training samples, arpack may be more efficient
    than the dense eigensolver.

tol : float, default=0
    Convergence tolerance for arpack.
    If 0, optimal value will be chosen by arpack.

max_iter : int, default=None
    Maximum number of iterations for arpack.
    If None, optimal value will be chosen by arpack.

remove_zero_eig : boolean, default=False
    If True, then all components with zero eigenvalues are removed, so
    that the number of components in the output may be < n_components
    (and sometimes even zero due to numerical instability).
    When n_components is None, this parameter is ignored and components
    with zero eigenvalues are removed regardless.

random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`. Used when ``eigen_solver`` == 'arpack'.

    .. versionadded:: 0.18

copy_X : boolean, default=True
    If True, input X is copied and stored by the model in the `X_fit_`
    attribute. If no further changes will be done to X, setting
    `copy_X=False` saves memory by storing a reference.

    .. versionadded:: 0.18

n_jobs : int or None, optional (default=None)
    The number of parallel jobs to run.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

    .. versionadded:: 0.18

Attributes
----------
lambdas_ : array, (n_components,)
    Eigenvalues of the centered kernel matrix in decreasing order.
    If `n_components` and `remove_zero_eig` are not set,
    then all values are stored.

alphas_ : array, (n_samples, n_components)
    Eigenvectors of the centered kernel matrix. If `n_components` and
    `remove_zero_eig` are not set, then all components are stored.

dual_coef_ : array, (n_samples, n_features)
    Inverse transform matrix. Only available when
    ``fit_inverse_transform`` is True.

X_transformed_fit_ : array, (n_samples, n_components)
    Projection of the fitted data on the kernel principal components.
    Only available when ``fit_inverse_transform`` is True.

X_fit_ : (n_samples, n_features)
    The data used to fit the model. If `copy_X=False`, then `X_fit_` is
    a reference. This attribute is used for the calls to transform.

Examples
--------
>>> from sklearn.datasets import load_digits
>>> from sklearn.decomposition import KernelPCA
>>> X, _ = load_digits(return_X_y=True)
>>> transformer = KernelPCA(n_components=7, kernel='linear')
>>> X_transformed = transformer.fit_transform(X)
>>> X_transformed.shape
(1797, 7)

References
----------
Kernel PCA was introduced in:
    Bernhard Schoelkopf, Alexander J. Smola,
    and Klaus-Robert Mueller. 1999. Kernel principal
    component analysis. In Advances in kernel methods,
    MIT Press, Cambridge, MA, USA 327-352.
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/decomposition/_kernel_pca.py#L265)
> `fit(self, X, y=None)`

Fit the model from data in X.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training vector, where n_samples in the number of samples
    and n_features is the number of features.

Returns
-------
self : object
    Returns the instance itself.
### `fit_transform`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/decomposition/_kernel_pca.py#L293)
> `fit_transform(self, X, y=None, **params)`

Fit the model from data in X and transform X.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training vector, where n_samples in the number of samples
    and n_features is the number of features.

Returns
-------
X_new : array-like, shape (n_samples, n_components)
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
### `inverse_transform`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/decomposition/_kernel_pca.py#L341)
> `inverse_transform(self, X)`

Transform X back to original space.

Parameters
----------
X : array-like, shape (n_samples, n_components)

Returns
-------
X_new : array-like, shape (n_samples, n_features)

References
----------
"Learning to Find Pre-Images", G BakIr et al, 2004.
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L233)
> `run(self, input)`

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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/decomposition/_kernel_pca.py#L316)
> `transform(self, X)`

Transform X.

Parameters
----------
X : array-like, shape (n_samples, n_features)

Returns
-------
X_new : array-like, shape (n_samples, n_components)
