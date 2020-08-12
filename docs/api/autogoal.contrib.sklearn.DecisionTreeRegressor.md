# `autogoal.contrib.sklearn.DecisionTreeRegressor`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1810)
> `DecisionTreeRegressor(self, min_samples_split, min_weight_fraction_leaf, min_impurity_decrease, ccp_alpha)`

A decision tree regressor.

Read more in the :ref:`User Guide <tree>`.

Parameters
----------
criterion : {"mse", "friedman_mse", "mae"}, default="mse"
    The function to measure the quality of a split. Supported criteria
    are "mse" for the mean squared error, which is equal to variance
    reduction as feature selection criterion and minimizes the L2 loss
    using the mean of each terminal node, "friedman_mse", which uses mean
    squared error with Friedman's improvement score for potential splits,
    and "mae" for the mean absolute error, which minimizes the L1 loss
    using the median of each terminal node.

    .. versionadded:: 0.18
       Mean Absolute Error (MAE) criterion.

splitter : {"best", "random"}, default="best"
    The strategy used to choose the split at each node. Supported
    strategies are "best" to choose the best split and "random" to choose
    the best random split.

max_depth : int, default=None
    The maximum depth of the tree. If None, then nodes are expanded until
    all leaves are pure or until all leaves contain less than
    min_samples_split samples.

min_samples_split : int or float, default=2
    The minimum number of samples required to split an internal node:

    - If int, then consider `min_samples_split` as the minimum number.
    - If float, then `min_samples_split` is a fraction and
      `ceil(min_samples_split * n_samples)` are the minimum
      number of samples for each split.

    .. versionchanged:: 0.18
       Added float values for fractions.

min_samples_leaf : int or float, default=1
    The minimum number of samples required to be at a leaf node.
    A split point at any depth will only be considered if it leaves at
    least ``min_samples_leaf`` training samples in each of the left and
    right branches.  This may have the effect of smoothing the model,
    especially in regression.

    - If int, then consider `min_samples_leaf` as the minimum number.
    - If float, then `min_samples_leaf` is a fraction and
      `ceil(min_samples_leaf * n_samples)` are the minimum
      number of samples for each node.

    .. versionchanged:: 0.18
       Added float values for fractions.

min_weight_fraction_leaf : float, default=0.0
    The minimum weighted fraction of the sum total of weights (of all
    the input samples) required to be at a leaf node. Samples have
    equal weight when sample_weight is not provided.

max_features : int, float or {"auto", "sqrt", "log2"}, default=None
    The number of features to consider when looking for the best split:

    - If int, then consider `max_features` features at each split.
    - If float, then `max_features` is a fraction and
      `int(max_features * n_features)` features are considered at each
      split.
    - If "auto", then `max_features=n_features`.
    - If "sqrt", then `max_features=sqrt(n_features)`.
    - If "log2", then `max_features=log2(n_features)`.
    - If None, then `max_features=n_features`.

    Note: the search for a split does not stop until at least one
    valid partition of the node samples is found, even if it requires to
    effectively inspect more than ``max_features`` features.

random_state : int or RandomState, default=None
    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.

max_leaf_nodes : int, default=None
    Grow a tree with ``max_leaf_nodes`` in best-first fashion.
    Best nodes are defined as relative reduction in impurity.
    If None then unlimited number of leaf nodes.

min_impurity_decrease : float, default=0.0
    A node will be split if this split induces a decrease of the impurity
    greater than or equal to this value.

    The weighted impurity decrease equation is the following::

        N_t / N * (impurity - N_t_R / N_t * right_impurity
                            - N_t_L / N_t * left_impurity)

    where ``N`` is the total number of samples, ``N_t`` is the number of
    samples at the current node, ``N_t_L`` is the number of samples in the
    left child, and ``N_t_R`` is the number of samples in the right child.

    ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
    if ``sample_weight`` is passed.

    .. versionadded:: 0.19

min_impurity_split : float, (default=1e-7)
    Threshold for early stopping in tree growth. A node will split
    if its impurity is above the threshold, otherwise it is a leaf.

    .. deprecated:: 0.19
       ``min_impurity_split`` has been deprecated in favor of
       ``min_impurity_decrease`` in 0.19. The default value of
       ``min_impurity_split`` will change from 1e-7 to 0 in 0.23 and it
       will be removed in 0.25. Use ``min_impurity_decrease`` instead.

presort : deprecated, default='deprecated'
    This parameter is deprecated and will be removed in v0.24.

    .. deprecated:: 0.22

ccp_alpha : non-negative float, default=0.0
    Complexity parameter used for Minimal Cost-Complexity Pruning. The
    subtree with the largest cost complexity that is smaller than
    ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
    :ref:`minimal_cost_complexity_pruning` for details.

    .. versionadded:: 0.22

Attributes
----------
feature_importances_ : ndarray of shape (n_features,)
    The feature importances.
    The higher, the more important the feature.
    The importance of a feature is computed as the
    (normalized) total reduction of the criterion brought
    by that feature. It is also known as the Gini importance [4]_.

max_features_ : int
    The inferred value of max_features.

n_features_ : int
    The number of features when ``fit`` is performed.

n_outputs_ : int
    The number of outputs when ``fit`` is performed.

tree_ : Tree
    The underlying Tree object. Please refer to
    ``help(sklearn.tree._tree.Tree)`` for attributes of Tree object and
    :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
    for basic usage of these attributes.

See Also
--------
DecisionTreeClassifier : A decision tree classifier.

Notes
-----
The default values for the parameters controlling the size of the trees
(e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
unpruned trees which can potentially be very large on some data sets. To
reduce memory consumption, the complexity and size of the trees should be
controlled by setting those parameter values.

The features are always randomly permuted at each split. Therefore,
the best found split may vary, even with the same training data and
``max_features=n_features``, if the improvement of the criterion is
identical for several splits enumerated during the search of the best
split. To obtain a deterministic behaviour during fitting,
``random_state`` has to be fixed.

References
----------

.. [1] https://en.wikipedia.org/wiki/Decision_tree_learning

.. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
       and Regression Trees", Wadsworth, Belmont, CA, 1984.

.. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
       Learning", Springer, 2009.

.. [4] L. Breiman, and A. Cutler, "Random Forests",
       https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

Examples
--------
>>> from sklearn.datasets import load_boston
>>> from sklearn.model_selection import cross_val_score
>>> from sklearn.tree import DecisionTreeRegressor
>>> X, y = load_boston(return_X_y=True)
>>> regressor = DecisionTreeRegressor(random_state=0)
>>> cross_val_score(regressor, X, y, cv=10)
...                    # doctest: +SKIP
...
array([ 0.61..., 0.57..., -0.34..., 0.41..., 0.75...,
        0.07..., 0.29..., 0.33..., -1.42..., -1.77...])
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `apply`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/tree/_classes.py#L447)
> `apply(self, X, check_input=True)`

Return the index of the leaf that each sample is predicted as.

.. versionadded:: 0.17

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csr_matrix``.

check_input : bool, default=True
    Allow to bypass several input checking.
    Don't use this parameter unless you know what you do.

Returns
-------
X_leaves : array-like of shape (n_samples,)
    For each datapoint x in X, return the index of the leaf x
    ends up in. Leaves are numbered within
    ``[0; self.tree_.node_count)``, possibly with gaps in the
    numbering.
### `cost_complexity_pruning_path`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/tree/_classes.py#L523)
> `cost_complexity_pruning_path(self, X, y, sample_weight=None)`

Compute the pruning path during Minimal Cost-Complexity Pruning.

See :ref:`minimal_cost_complexity_pruning` for details on the pruning
process.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The training input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csc_matrix``.

y : array-like of shape (n_samples,) or (n_samples, n_outputs)
    The target values (class labels) as integers or strings.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights. If None, then samples are equally weighted. Splits
    that would create child nodes with net zero or negative weight are
    ignored while searching for a split in each node. Splits are also
    ignored if they would result in any single class carrying a
    negative weight in either child node.

Returns
-------
ccp_path : Bunch
    Dictionary-like object, with attributes:

    ccp_alphas : ndarray
        Effective alphas of subtree during pruning.

    impurities : ndarray
        Sum of the impurities of the subtree leaves for the
        corresponding alpha value in ``ccp_alphas``.
### `decision_path`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/tree/_classes.py#L475)
> `decision_path(self, X, check_input=True)`

Return the decision path in the tree.

.. versionadded:: 0.18

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csr_matrix``.

check_input : bool, default=True
    Allow to bypass several input checking.
    Don't use this parameter unless you know what you do.

Returns
-------
indicator : sparse matrix of shape (n_samples, n_nodes)
    Return a node indicator CSR matrix where non zero elements
    indicates that the samples goes through the nodes.
### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/tree/_classes.py#L1184)
> `fit(self, X, y, sample_weight=None, check_input=True, X_idx_sorted=None)`

Build a decision tree regressor from the training set (X, y).

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The training input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csc_matrix``.

y : array-like of shape (n_samples,) or (n_samples, n_outputs)
    The target values (real numbers). Use ``dtype=np.float64`` and
    ``order='C'`` for maximum efficiency.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights. If None, then samples are equally weighted. Splits
    that would create child nodes with net zero or negative weight are
    ignored while searching for a split in each node.

check_input : bool, default=True
    Allow to bypass several input checking.
    Don't use this parameter unless you know what you do.

X_idx_sorted : array-like of shape (n_samples, n_features),             default=None
    The indexes of the sorted training input samples. If many tree
    are grown on the same dataset, this allows the ordering to be
    cached between trees. If None, the data will be sorted here.
    Don't use this parameter unless you know what to do.

Returns
-------
self : DecisionTreeRegressor
    Fitted estimator.
### `get_depth`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/tree/_classes.py#L115)
> `get_depth(self)`

Return the depth of the decision tree.

The depth of a tree is the maximum distance between the root
and any leaf.

Returns
-------
self.tree_.max_depth : int
    The maximum depth of the tree.
### `get_n_leaves`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/tree/_classes.py#L129)
> `get_n_leaves(self)`

Return the number of leaves of the decision tree.

Returns
-------
self.tree_.n_leaves : int
    Number of leaves.
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/tree/_classes.py#L395)
> `predict(self, X, check_input=True)`

Predict class or regression value for X.

For a classification model, the predicted class for each sample in X is
returned. For a regression model, the predicted value based on X is
returned.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    The input samples. Internally, it will be converted to
    ``dtype=np.float32`` and if a sparse matrix is provided
    to a sparse ``csr_matrix``.

check_input : bool, default=True
    Allow to bypass several input checking.
    Don't use this parameter unless you know what you do.

Returns
-------
y : array-like of shape (n_samples,) or (n_samples, n_outputs)
    The predicted classes, or the predict values.
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1827)
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

