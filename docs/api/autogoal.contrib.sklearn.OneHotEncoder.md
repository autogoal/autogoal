# `autogoal.contrib.sklearn.OneHotEncoder`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1502)
> `OneHotEncoder(self, categories, sparse, handle_unknown)`

Encode categorical features as a one-hot numeric array.

The input to this transformer should be an array-like of integers or
strings, denoting the values taken on by categorical (discrete) features.
The features are encoded using a one-hot (aka 'one-of-K' or 'dummy')
encoding scheme. This creates a binary column for each category and
returns a sparse matrix or dense array (depending on the ``sparse``
parameter)

By default, the encoder derives the categories based on the unique values
in each feature. Alternatively, you can also specify the `categories`
manually.

This encoding is needed for feeding categorical data to many scikit-learn
estimators, notably linear models and SVMs with the standard kernels.

Note: a one-hot encoding of y labels should use a LabelBinarizer
instead.

Read more in the :ref:`User Guide <preprocessing_categorical_features>`.

.. versionchanged:: 0.20

Parameters
----------
categories : 'auto' or a list of array-like, default='auto'
    Categories (unique values) per feature:

    - 'auto' : Determine categories automatically from the training data.
    - list : ``categories[i]`` holds the categories expected in the ith
      column. The passed categories should not mix strings and numeric
      values within a single feature, and should be sorted in case of
      numeric values.

    The used categories can be found in the ``categories_`` attribute.

drop : 'first' or a array-like of shape (n_features,), default=None
    Specifies a methodology to use to drop one of the categories per
    feature. This is useful in situations where perfectly collinear
    features cause problems, such as when feeding the resulting data
    into a neural network or an unregularized regression.

    - None : retain all features (the default).
    - 'first' : drop the first category in each feature. If only one
      category is present, the feature will be dropped entirely.
    - array : ``drop[i]`` is the category in feature ``X[:, i]`` that
      should be dropped.

sparse : bool, default=True
    Will return sparse matrix if set True else will return an array.

dtype : number type, default=np.float
    Desired dtype of output.

handle_unknown : {'error', 'ignore'}, default='error'
    Whether to raise an error or ignore if an unknown categorical feature
    is present during transform (default is to raise). When this parameter
    is set to 'ignore' and an unknown category is encountered during
    transform, the resulting one-hot encoded columns for this feature
    will be all zeros. In the inverse transform, an unknown category
    will be denoted as None.

Attributes
----------
categories_ : list of arrays
    The categories of each feature determined during fitting
    (in order of the features in X and corresponding with the output
    of ``transform``). This includes the category specified in ``drop``
    (if any).

drop_idx_ : array of shape (n_features,)
    ``drop_idx_[i]`` is the index in ``categories_[i]`` of the category to
    be dropped for each feature. None if all the transformed features will
    be retained.

See Also
--------
sklearn.preprocessing.OrdinalEncoder : Performs an ordinal (integer)
  encoding of the categorical features.
sklearn.feature_extraction.DictVectorizer : Performs a one-hot encoding of
  dictionary items (also handles string-valued features).
sklearn.feature_extraction.FeatureHasher : Performs an approximate one-hot
  encoding of dictionary items or strings.
sklearn.preprocessing.LabelBinarizer : Binarizes labels in a one-vs-all
  fashion.
sklearn.preprocessing.MultiLabelBinarizer : Transforms between iterable of
  iterables and a multilabel format, e.g. a (samples x classes) binary
  matrix indicating the presence of a class label.

Examples
--------
Given a dataset with two features, we let the encoder find the unique
values per feature and transform the data to a binary one-hot encoding.

>>> from sklearn.preprocessing import OneHotEncoder
>>> enc = OneHotEncoder(handle_unknown='ignore')
>>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
>>> enc.fit(X)
OneHotEncoder(handle_unknown='ignore')
>>> enc.categories_
[array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
>>> enc.transform([['Female', 1], ['Male', 4]]).toarray()
array([[1., 0., 1., 0., 0.],
       [0., 1., 0., 0., 0.]])
>>> enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]])
array([['Male', 1],
       [None, 2]], dtype=object)
>>> enc.get_feature_names(['gender', 'group'])
array(['gender_Female', 'gender_Male', 'group_1', 'group_2', 'group_3'],
  dtype=object)
>>> drop_enc = OneHotEncoder(drop='first').fit(X)
>>> drop_enc.categories_
[array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
>>> drop_enc.transform([['Female', 1], ['Male', 2]]).toarray()
array([[0., 0., 0.],
       [1., 1., 0.]])
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/preprocessing/_encoders.py#L329)
> `fit(self, X, y=None)`

Fit OneHotEncoder to X.

Parameters
----------
X : array-like, shape [n_samples, n_features]
    The data to determine the categories of each feature.

y : None
    Ignored. This parameter exists only for compatibility with
    :class:`sklearn.pipeline.Pipeline`.

Returns
-------
self
### `fit_transform`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/preprocessing/_encoders.py#L351)
> `fit_transform(self, X, y=None)`

Fit OneHotEncoder to X, then transform X.

Equivalent to fit(X).transform(X) but more convenient.

Parameters
----------
X : array-like, shape [n_samples, n_features]
    The data to encode.

y : None
    Ignored. This parameter exists only for compatibility with
    :class:`sklearn.pipeline.Pipeline`.

Returns
-------
X_out : sparse matrix if sparse=True else a 2-d array
    Transformed input.
### `get_feature_names`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/preprocessing/_encoders.py#L509)
> `get_feature_names(self, input_features=None)`

Return feature names for output features.

Parameters
----------
input_features : list of str of shape (n_features,)
    String names for input features if available. By default,
    "x0", "x1", ... "xn_features" is used.

Returns
-------
output_feature_names : ndarray of shape (n_output_features,)
    Array of feature names.
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/preprocessing/_encoders.py#L423)
> `inverse_transform(self, X)`

Convert the data back to the original representation.

In case unknown categories are encountered (all zeros in the
one-hot encoding), ``None`` is used to represent this category.

Parameters
----------
X : array-like or sparse matrix, shape [n_samples, n_encoded_features]
    The transformed data.

Returns
-------
X_tr : array-like, shape [n_samples, n_features]
    Inverse transformed array.
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1514)
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/preprocessing/_encoders.py#L374)
> `transform(self, X)`

Transform X using one-hot encoding.

Parameters
----------
X : array-like, shape [n_samples, n_features]
    The data to encode.

Returns
-------
X_out : sparse matrix if sparse=True else a 2-d array
    Transformed input.
