# `autogoal.contrib.sklearn.LabelBinarizer`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1535)
> `LabelBinarizer(self, sparse_output)`

Binarize labels in a one-vs-all fashion

Several regression and binary classification algorithms are
available in scikit-learn. A simple way to extend these algorithms
to the multi-class classification case is to use the so-called
one-vs-all scheme.

At learning time, this simply consists in learning one regressor
or binary classifier per class. In doing so, one needs to convert
multi-class labels to binary labels (belong or does not belong
to the class). LabelBinarizer makes this process easy with the
transform method.

At prediction time, one assigns the class for which the corresponding
model gave the greatest confidence. LabelBinarizer makes this easy
with the inverse_transform method.

Read more in the :ref:`User Guide <preprocessing_targets>`.

Parameters
----------

neg_label : int (default: 0)
    Value with which negative labels must be encoded.

pos_label : int (default: 1)
    Value with which positive labels must be encoded.

sparse_output : boolean (default: False)
    True if the returned array from transform is desired to be in sparse
    CSR format.

Attributes
----------

classes_ : array of shape [n_class]
    Holds the label for each class.

y_type_ : str,
    Represents the type of the target data as evaluated by
    utils.multiclass.type_of_target. Possible type are 'continuous',
    'continuous-multioutput', 'binary', 'multiclass',
    'multiclass-multioutput', 'multilabel-indicator', and 'unknown'.

sparse_input_ : boolean,
    True if the input data to transform is given as a sparse matrix, False
    otherwise.

Examples
--------
>>> from sklearn import preprocessing
>>> lb = preprocessing.LabelBinarizer()
>>> lb.fit([1, 2, 6, 4, 2])
LabelBinarizer()
>>> lb.classes_
array([1, 2, 4, 6])
>>> lb.transform([1, 6])
array([[1, 0, 0, 0],
       [0, 0, 0, 1]])

Binary targets transform to a column vector

>>> lb = preprocessing.LabelBinarizer()
>>> lb.fit_transform(['yes', 'no', 'no', 'yes'])
array([[1],
       [0],
       [0],
       [1]])

Passing a 2D matrix for multilabel classification

>>> import numpy as np
>>> lb.fit(np.array([[0, 1, 1], [1, 0, 0]]))
LabelBinarizer()
>>> lb.classes_
array([0, 1, 2])
>>> lb.transform([0, 1, 2, 1])
array([[1, 0, 0],
       [0, 1, 0],
       [0, 0, 1],
       [0, 1, 0]])

See also
--------
label_binarize : function to perform the transform operation of
    LabelBinarizer with fixed classes.
sklearn.preprocessing.OneHotEncoder : encode categorical features
    using a one-hot aka one-of-K scheme.
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/preprocessing/_label.py#L411)
> `fit(self, y)`

Fit label binarizer

Parameters
----------
y : array of shape [n_samples,] or [n_samples, n_classes]
    Target values. The 2-d matrix should only contain 0 and 1,
    represents multilabel classification.

Returns
-------
self : returns an instance of self.
### `fit_transform`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/preprocessing/_label.py#L435)
> `fit_transform(self, y)`

Fit label binarizer and transform multi-class labels to binary
labels.

The output of transform is sometimes referred to as
the 1-of-K coding scheme.

Parameters
----------
y : array or sparse matrix of shape [n_samples,] or             [n_samples, n_classes]
    Target values. The 2-d matrix should only contain 0 and 1,
    represents multilabel classification. Sparse matrix can be
    CSR, CSC, COO, DOK, or LIL.

Returns
-------
Y : array or CSR matrix of shape [n_samples, n_classes]
    Shape will be [n_samples, 1] for binary problems.
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/preprocessing/_label.py#L488)
> `inverse_transform(self, Y, threshold=None)`

Transform binary labels back to multi-class labels

Parameters
----------
Y : numpy array or sparse matrix with shape [n_samples, n_classes]
    Target values. All sparse matrices are converted to CSR before
    inverse transformation.

threshold : float or None
    Threshold used in the binary and multi-label cases.

    Use 0 when ``Y`` contains the output of decision_function
    (classifier).
    Use 0.5 when ``Y`` contains the output of predict_proba.

    If None, the threshold is assumed to be half way between
    neg_label and pos_label.

Returns
-------
y : numpy array or CSR matrix of shape [n_samples] Target values.

Notes
-----
In the case when the binary labels are fractional
(probabilistic), inverse_transform chooses the class with the
greatest value. Typically, this allows to use the output of a
linear model's decision_function method directly as the input
of inverse_transform.
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L1540)
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/preprocessing/_label.py#L457)
> `transform(self, y)`

Transform multi-class labels to binary labels

The output of transform is sometimes referred to by some authors as
the 1-of-K coding scheme.

Parameters
----------
y : array or sparse matrix of shape [n_samples,] or             [n_samples, n_classes]
    Target values. The 2-d matrix should only contain 0 and 1,
    represents multilabel classification. Sparse matrix can be
    CSR, CSC, COO, DOK, or LIL.

Returns
-------
Y : numpy array or CSR matrix of shape [n_samples, n_classes]
    Shape will be [n_samples, 1] for binary problems.
