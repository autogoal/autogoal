# `autogoal.contrib.sklearn.LatentDirichletAllocation`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L243)
> `LatentDirichletAllocation(self)`

Latent Dirichlet Allocation with online variational Bayes algorithm

.. versionadded:: 0.17

Read more in the :ref:`User Guide <LatentDirichletAllocation>`.

Parameters
----------
n_components : int, optional (default=10)
    Number of topics.

doc_topic_prior : float, optional (default=None)
    Prior of document topic distribution `theta`. If the value is None,
    defaults to `1 / n_components`.
    In [1]_, this is called `alpha`.

topic_word_prior : float, optional (default=None)
    Prior of topic word distribution `beta`. If the value is None, defaults
    to `1 / n_components`.
    In [1]_, this is called `eta`.

learning_method : 'batch' | 'online', default='batch'
    Method used to update `_component`. Only used in :meth:`fit` method.
    In general, if the data size is large, the online update will be much
    faster than the batch update.

    Valid options::

        'batch': Batch variational Bayes method. Use all training data in
            each EM update.
            Old `components_` will be overwritten in each iteration.
        'online': Online variational Bayes method. In each EM update, use
            mini-batch of training data to update the ``components_``
            variable incrementally. The learning rate is controlled by the
            ``learning_decay`` and the ``learning_offset`` parameters.

    .. versionchanged:: 0.20
        The default learning method is now ``"batch"``.

learning_decay : float, optional (default=0.7)
    It is a parameter that control learning rate in the online learning
    method. The value should be set between (0.5, 1.0] to guarantee
    asymptotic convergence. When the value is 0.0 and batch_size is
    ``n_samples``, the update method is same as batch learning. In the
    literature, this is called kappa.

learning_offset : float, optional (default=10.)
    A (positive) parameter that downweights early iterations in online
    learning.  It should be greater than 1.0. In the literature, this is
    called tau_0.

max_iter : integer, optional (default=10)
    The maximum number of iterations.

batch_size : int, optional (default=128)
    Number of documents to use in each EM iteration. Only used in online
    learning.

evaluate_every : int, optional (default=0)
    How often to evaluate perplexity. Only used in `fit` method.
    set it to 0 or negative number to not evaluate perplexity in
    training at all. Evaluating perplexity can help you check convergence
    in training process, but it will also increase total training time.
    Evaluating perplexity in every iteration might increase training time
    up to two-fold.

total_samples : int, optional (default=1e6)
    Total number of documents. Only used in the :meth:`partial_fit` method.

perp_tol : float, optional (default=1e-1)
    Perplexity tolerance in batch learning. Only used when
    ``evaluate_every`` is greater than 0.

mean_change_tol : float, optional (default=1e-3)
    Stopping tolerance for updating document topic distribution in E-step.

max_doc_update_iter : int (default=100)
    Max number of iterations for updating document topic distribution in
    the E-step.

n_jobs : int or None, optional (default=None)
    The number of jobs to use in the E-step.
    ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
    for more details.

verbose : int, optional (default=0)
    Verbosity level.

random_state : int, RandomState instance or None, optional (default=None)
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

Attributes
----------
components_ : array, [n_components, n_features]
    Variational parameters for topic word distribution. Since the complete
    conditional for topic word distribution is a Dirichlet,
    ``components_[i, j]`` can be viewed as pseudocount that represents the
    number of times word `j` was assigned to topic `i`.
    It can also be viewed as distribution over the words for each topic
    after normalization:
    ``model.components_ / model.components_.sum(axis=1)[:, np.newaxis]``.

n_batch_iter_ : int
    Number of iterations of the EM step.

n_iter_ : int
    Number of passes over the dataset.

bound_ : float
    Final perplexity score on training set.

doc_topic_prior_ : float
    Prior of document topic distribution `theta`. If the value is None,
    it is `1 / n_components`.

topic_word_prior_ : float
    Prior of topic word distribution `beta`. If the value is None, it is
    `1 / n_components`.

Examples
--------
>>> from sklearn.decomposition import LatentDirichletAllocation
>>> from sklearn.datasets import make_multilabel_classification
>>> # This produces a feature matrix of token counts, similar to what
>>> # CountVectorizer would produce on text.
>>> X, _ = make_multilabel_classification(random_state=0)
>>> lda = LatentDirichletAllocation(n_components=5,
...     random_state=0)
>>> lda.fit(X)
LatentDirichletAllocation(...)
>>> # get topics for some given samples:
>>> lda.transform(X[-2:])
array([[0.00360392, 0.25499205, 0.0036211 , 0.64236448, 0.09541846],
       [0.15297572, 0.00362644, 0.44412786, 0.39568399, 0.003586  ]])

References
----------
.. [1] "Online Learning for Latent Dirichlet Allocation", Matthew D.
    Hoffman, David M. Blei, Francis Bach, 2010

[2] "Stochastic Variational Inference", Matthew D. Hoffman, David M. Blei,
    Chong Wang, John Paisley, 2013

[3] Matthew D. Hoffman's onlineldavb code. Link:
    https://github.com/blei-lab/onlineldavb
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/decomposition/_lda.py#L527)
> `fit(self, X, y=None)`

Learn model for the data X with variational Bayes method.

When `learning_method` is 'online', use mini-batch update.
Otherwise, use batch update.

Parameters
----------
X : array-like or sparse matrix, shape=(n_samples, n_features)
    Document word matrix.

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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/decomposition/_lda.py#L486)
> `partial_fit(self, X, y=None)`

Online VB with Mini-Batch update.

Parameters
----------
X : array-like or sparse matrix, shape=(n_samples, n_features)
    Document word matrix.

y : Ignored

Returns
-------
self
### `perplexity`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/decomposition/_lda.py#L792)
> `perplexity(self, X, sub_sampling=False)`

Calculate approximate perplexity for data X.

Perplexity is defined as exp(-1. * log-likelihood per word)

.. versionchanged:: 0.19
   *doc_topic_distr* argument has been deprecated and is ignored
   because user no longer has access to unnormalized distribution

Parameters
----------
X : array-like or sparse matrix, [n_samples, n_features]
    Document word matrix.

sub_sampling : bool
    Do sub-sampling or not.

Returns
-------
score : float
    Perplexity score.
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L248)
> `run(self, input)`

### `score`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/decomposition/_lda.py#L723)
> `score(self, X, y=None)`

Calculate approximate log-likelihood as score.

Parameters
----------
X : array-like or sparse matrix, shape=(n_samples, n_features)
    Document word matrix.

y : Ignored

Returns
-------
score : float
    Use approximate bound as score.
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/decomposition/_lda.py#L627)
> `transform(self, X)`

Transform data X according to the fitted model.

   .. versionchanged:: 0.18
      *doc_topic_distr* is now normalized

Parameters
----------
X : array-like or sparse matrix, shape=(n_samples, n_features)
    Document word matrix.

Returns
-------
doc_topic_distr : shape=(n_samples, n_components)
    Document topic distribution for X.
