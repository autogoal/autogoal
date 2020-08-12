# `autogoal.contrib.sklearn.TfidfTransformer`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L401)
> `TfidfTransformer(self, norm, use_idf, smooth_idf, sublinear_tf)`

Transform a count matrix to a normalized tf or tf-idf representation

Tf means term-frequency while tf-idf means term-frequency times inverse
document-frequency. This is a common term weighting scheme in information
retrieval, that has also found good use in document classification.

The goal of using tf-idf instead of the raw frequencies of occurrence of a
token in a given document is to scale down the impact of tokens that occur
very frequently in a given corpus and that are hence empirically less
informative than features that occur in a small fraction of the training
corpus.

The formula that is used to compute the tf-idf for a term t of a document d
in a document set is tf-idf(t, d) = tf(t, d) * idf(t), and the idf is
computed as idf(t) = log [ n / df(t) ] + 1 (if ``smooth_idf=False``), where
n is the total number of documents in the document set and df(t) is the
document frequency of t; the document frequency is the number of documents
in the document set that contain the term t. The effect of adding "1" to
the idf in the equation above is that terms with zero idf, i.e., terms
that occur in all documents in a training set, will not be entirely
ignored.
(Note that the idf formula above differs from the standard textbook
notation that defines the idf as
idf(t) = log [ n / (df(t) + 1) ]).

If ``smooth_idf=True`` (the default), the constant "1" is added to the
numerator and denominator of the idf as if an extra document was seen
containing every term in the collection exactly once, which prevents
zero divisions: idf(d, t) = log [ (1 + n) / (1 + df(d, t)) ] + 1.

Furthermore, the formulas used to compute tf and idf depend
on parameter settings that correspond to the SMART notation used in IR
as follows:

Tf is "n" (natural) by default, "l" (logarithmic) when
``sublinear_tf=True``.
Idf is "t" when use_idf is given, "n" (none) otherwise.
Normalization is "c" (cosine) when ``norm='l2'``, "n" (none)
when ``norm=None``.

Read more in the :ref:`User Guide <text_feature_extraction>`.

Parameters
----------
norm : 'l1', 'l2' or None, optional (default='l2')
    Each output row will have unit norm, either:
    * 'l2': Sum of squares of vector elements is 1. The cosine
    similarity between two vectors is their dot product when l2 norm has
    been applied.
    * 'l1': Sum of absolute values of vector elements is 1.
    See :func:`preprocessing.normalize`

use_idf : boolean (default=True)
    Enable inverse-document-frequency reweighting.

smooth_idf : boolean (default=True)
    Smooth idf weights by adding one to document frequencies, as if an
    extra document was seen containing every term in the collection
    exactly once. Prevents zero divisions.

sublinear_tf : boolean (default=False)
    Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

Attributes
----------
idf_ : array, shape (n_features)
    The inverse document frequency (IDF) vector; only defined
    if  ``use_idf`` is True.

Examples
--------
>>> from sklearn.feature_extraction.text import TfidfTransformer
>>> from sklearn.feature_extraction.text import CountVectorizer
>>> from sklearn.pipeline import Pipeline
>>> import numpy as np
>>> corpus = ['this is the first document',
...           'this document is the second document',
...           'and this is the third one',
...           'is this the first document']
>>> vocabulary = ['this', 'document', 'first', 'is', 'second', 'the',
...               'and', 'one']
>>> pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)),
...                  ('tfid', TfidfTransformer())]).fit(corpus)
>>> pipe['count'].transform(corpus).toarray()
array([[1, 1, 1, 1, 0, 1, 0, 0],
       [1, 2, 0, 1, 1, 1, 0, 0],
       [1, 0, 0, 1, 0, 1, 1, 1],
       [1, 1, 1, 1, 0, 1, 0, 0]])
>>> pipe['tfid'].idf_
array([1.        , 1.22314355, 1.51082562, 1.        , 1.91629073,
       1.        , 1.91629073, 1.91629073])
>>> pipe.transform(corpus).shape
(4, 8)

References
----------

.. [Yates2011] R. Baeza-Yates and B. Ribeiro-Neto (2011). Modern
               Information Retrieval. Addison Wesley, pp. 68-74.

.. [MRS2008] C.D. Manning, P. Raghavan and H. Schütze  (2008).
               Introduction to Information Retrieval. Cambridge University
               Press, pp. 118-120.
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/feature_extraction/text.py#L1442)
> `fit(self, X, y=None)`

Learn the idf vector (global term weights)

Parameters
----------
X : sparse matrix, [n_samples, n_features]
    a matrix of term/token counts
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
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L418)
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/feature_extraction/text.py#L1474)
> `transform(self, X, copy=True)`

Transform a count matrix to a tf or tf-idf representation

Parameters
----------
X : sparse matrix, [n_samples, n_features]
    a matrix of term/token counts

copy : boolean, default True
    Whether to copy X and operate on the copy or perform in-place
    operations.

Returns
-------
vectors : sparse matrix, [n_samples, n_features]
