# `autogoal.contrib.sklearn.IncrementalPCA`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L199)
> `IncrementalPCA(self, whiten)`

Incremental principal components analysis (IPCA).

Linear dimensionality reduction using Singular Value Decomposition of
the data, keeping only the most significant singular vectors to
project the data to a lower dimensional space. The input data is centered
but not scaled for each feature before applying the SVD.

Depending on the size of the input data, this algorithm can be much more
memory efficient than a PCA, and allows sparse input.

This algorithm has constant memory complexity, on the order
of ``batch_size * n_features``, enabling use of np.memmap files without
loading the entire file into memory. For sparse matrices, the input
is converted to dense in batches (in order to be able to subtract the
mean) which avoids storing the entire dense matrix at any one time.

The computational overhead of each SVD is
``O(batch_size * n_features ** 2)``, but only 2 * batch_size samples
remain in memory at a time. There will be ``n_samples / batch_size`` SVD
computations to get the principal components, versus 1 large SVD of
complexity ``O(n_samples * n_features ** 2)`` for PCA.

Read more in the :ref:`User Guide <IncrementalPCA>`.

.. versionadded:: 0.16

Parameters
----------
n_components : int or None, (default=None)
    Number of components to keep. If ``n_components `` is ``None``,
    then ``n_components`` is set to ``min(n_samples, n_features)``.

whiten : bool, optional
    When True (False by default) the ``components_`` vectors are divided
    by ``n_samples`` times ``components_`` to ensure uncorrelated outputs
    with unit component-wise variances.

    Whitening will remove some information from the transformed signal
    (the relative variance scales of the components) but can sometimes
    improve the predictive accuracy of the downstream estimators by
    making data respect some hard-wired assumptions.

copy : bool, (default=True)
    If False, X will be overwritten. ``copy=False`` can be used to
    save memory but is unsafe for general use.

batch_size : int or None, (default=None)
    The number of samples to use for each batch. Only used when calling
    ``fit``. If ``batch_size`` is ``None``, then ``batch_size``
    is inferred from the data and set to ``5 * n_features``, to provide a
    balance between approximation accuracy and memory consumption.

Attributes
----------
components_ : array, shape (n_components, n_features)
    Components with maximum variance.

explained_variance_ : array, shape (n_components,)
    Variance explained by each of the selected components.

explained_variance_ratio_ : array, shape (n_components,)
    Percentage of variance explained by each of the selected components.
    If all components are stored, the sum of explained variances is equal
    to 1.0.

singular_values_ : array, shape (n_components,)
    The singular values corresponding to each of the selected components.
    The singular values are equal to the 2-norms of the ``n_components``
    variables in the lower-dimensional space.

mean_ : array, shape (n_features,)
    Per-feature empirical mean, aggregate over calls to ``partial_fit``.

var_ : array, shape (n_features,)
    Per-feature empirical variance, aggregate over calls to
    ``partial_fit``.

noise_variance_ : float
    The estimated noise covariance following the Probabilistic PCA model
    from Tipping and Bishop 1999. See "Pattern Recognition and
    Machine Learning" by C. Bishop, 12.2.1 p. 574 or
    http://www.miketipping.com/papers/met-mppca.pdf.

n_components_ : int
    The estimated number of components. Relevant when
    ``n_components=None``.

n_samples_seen_ : int
    The number of samples processed by the estimator. Will be reset on
    new calls to fit, but increments across ``partial_fit`` calls.

Examples
--------
>>> from sklearn.datasets import load_digits
>>> from sklearn.decomposition import IncrementalPCA
>>> from scipy import sparse
>>> X, _ = load_digits(return_X_y=True)
>>> transformer = IncrementalPCA(n_components=7, batch_size=200)
>>> # either partially fit on smaller batches of data
>>> transformer.partial_fit(X[:100, :])
IncrementalPCA(batch_size=200, n_components=7)
>>> # or let the fit function itself divide the data into batches
>>> X_sparse = sparse.csr_matrix(X)
>>> X_transformed = transformer.fit_transform(X_sparse)
>>> X_transformed.shape
(1797, 7)

Notes
-----
Implements the incremental PCA model from:
*D. Ross, J. Lim, R. Lin, M. Yang, Incremental Learning for Robust Visual
Tracking, International Journal of Computer Vision, Volume 77, Issue 1-3,
pp. 125-141, May 2008.*
See https://www.cs.toronto.edu/~dross/ivt/RossLimLinYang_ijcv.pdf

This model is an extension of the Sequential Karhunen-Loeve Transform from:
*A. Levy and M. Lindenbaum, Sequential Karhunen-Loeve Basis Extraction and
its Application to Images, IEEE Transactions on Image Processing, Volume 9,
Number 8, pp. 1371-1374, August 2000.*
See https://www.cs.technion.ac.il/~mic/doc/skl-ip.pdf

We have specifically abstained from an optimization used by authors of both
papers, a QR decomposition used in specific situations to reduce the
algorithmic complexity of the SVD. The source for this technique is
*Matrix Computations, Third Edition, G. Holub and C. Van Loan, Chapter 5,
section 5.4.4, pp 252-253.*. This technique has been omitted because it is
advantageous only when decomposing a matrix with ``n_samples`` (rows)
>= 5/3 * ``n_features`` (columns), and hurts the readability of the
implemented algorithm. This would be a good opportunity for future
optimization, if it is deemed necessary.

References
----------
D. Ross, J. Lim, R. Lin, M. Yang. Incremental Learning for Robust Visual
Tracking, International Journal of Computer Vision, Volume 77,
Issue 1-3, pp. 125-141, May 2008.

G. Golub and C. Van Loan. Matrix Computations, Third Edition, Chapter 5,
Section 5.4.4, pp. 252-253.

See also
--------
PCA
KernelPCA
SparsePCA
TruncatedSVD
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/decomposition/_incremental_pca.py#L171)
> `fit(self, X, y=None)`

Fit the model with X, using minibatches of size batch_size.

Parameters
----------
X : array-like or sparse matrix, shape (n_samples, n_features)
    Training data, where n_samples is the number of samples and
    n_features is the number of features.

y : Ignored

Returns
-------
self : object
    Returns the instance itself.
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/decomposition/_base.py#L26)
> `get_covariance(self)`

Compute data covariance with the generative model.

``cov = components_.T * S**2 * components_ + sigma2 * eye(n_features)``
where S**2 contains the explained variances, and sigma2 contains the
noise variances.

Returns
-------
cov : array, shape=(n_features, n_features)
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/decomposition/_base.py#L47)
> `get_precision(self)`

Compute data precision matrix with the generative model.

Equals the inverse of the covariance but computed with
the matrix inversion lemma for efficiency.

Returns
-------
precision : array, shape=(n_features, n_features)
    Estimated precision of data.
### `inverse_transform`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/decomposition/_base.py#L135)
> `inverse_transform(self, X)`

Transform data back to its original space.

In other words, return an input X_original whose transform would be X.

Parameters
----------
X : array-like, shape (n_samples, n_components)
    New data, where n_samples is the number of samples
    and n_components is the number of components.

Returns
-------
X_original array-like, shape (n_samples, n_features)

Notes
-----
If whitening is enabled, inverse_transform will compute the
exact inverse operation, which includes reversing whitening.
### `partial_fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/decomposition/_incremental_pca.py#L215)
> `partial_fit(self, X, y=None, check_input=True)`

Incremental fit with X. All of X is processed as a single batch.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    Training data, where n_samples is the number of samples and
    n_features is the number of features.
check_input : bool
    Run check_array on X.

y : Ignored

Returns
-------
self : object
    Returns the instance itself.
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L204)
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/decomposition/_incremental_pca.py#L314)
> `transform(self, X)`

Apply dimensionality reduction to X.

X is projected on the first principal components previously extracted
from a training set, using minibatches of size batch_size if X is
sparse.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    New data, where n_samples is the number of samples
    and n_features is the number of features.

Returns
-------
X_new : array-like, shape (n_samples, n_components)

Examples
--------

>>> import numpy as np
>>> from sklearn.decomposition import IncrementalPCA
>>> X = np.array([[-1, -1], [-2, -1], [-3, -2],
...               [1, 1], [2, 1], [3, 2]])
>>> ipca = IncrementalPCA(n_components=2, batch_size=3)
>>> ipca.fit(X)
IncrementalPCA(batch_size=3, n_components=2)
>>> ipca.transform(X) # doctest: +SKIP
