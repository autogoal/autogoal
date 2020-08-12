# `autogoal.contrib.sklearn.TfidfVectorizer`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L426)
> `TfidfVectorizer(self, lowercase, binary, use_idf, smooth_idf, sublinear_tf)`

Convert a collection of raw documents to a matrix of TF-IDF features.

Equivalent to :class:`CountVectorizer` followed by
:class:`TfidfTransformer`.

Read more in the :ref:`User Guide <text_feature_extraction>`.

Parameters
----------
input : str {'filename', 'file', 'content'}
    If 'filename', the sequence passed as an argument to fit is
    expected to be a list of filenames that need reading to fetch
    the raw content to analyze.

    If 'file', the sequence items must have a 'read' method (file-like
    object) that is called to fetch the bytes in memory.

    Otherwise the input is expected to be a sequence of items that
    can be of type string or byte.

encoding : str, default='utf-8'
    If bytes or files are given to analyze, this encoding is used to
    decode.

decode_error : {'strict', 'ignore', 'replace'} (default='strict')
    Instruction on what to do if a byte sequence is given to analyze that
    contains characters not of the given `encoding`. By default, it is
    'strict', meaning that a UnicodeDecodeError will be raised. Other
    values are 'ignore' and 'replace'.

strip_accents : {'ascii', 'unicode', None} (default=None)
    Remove accents and perform other character normalization
    during the preprocessing step.
    'ascii' is a fast method that only works on characters that have
    an direct ASCII mapping.
    'unicode' is a slightly slower method that works on any characters.
    None (default) does nothing.

    Both 'ascii' and 'unicode' use NFKD normalization from
    :func:`unicodedata.normalize`.

lowercase : bool (default=True)
    Convert all characters to lowercase before tokenizing.

preprocessor : callable or None (default=None)
    Override the preprocessing (string transformation) stage while
    preserving the tokenizing and n-grams generation steps.
    Only applies if ``analyzer is not callable``.

tokenizer : callable or None (default=None)
    Override the string tokenization step while preserving the
    preprocessing and n-grams generation steps.
    Only applies if ``analyzer == 'word'``.

analyzer : str, {'word', 'char', 'char_wb'} or callable
    Whether the feature should be made of word or character n-grams.
    Option 'char_wb' creates character n-grams only from text inside
    word boundaries; n-grams at the edges of words are padded with space.

    If a callable is passed it is used to extract the sequence of features
    out of the raw, unprocessed input.

    .. versionchanged:: 0.21

    Since v0.21, if ``input`` is ``filename`` or ``file``, the data is
    first read from the file and then passed to the given callable
    analyzer.

stop_words : str {'english'}, list, or None (default=None)
    If a string, it is passed to _check_stop_list and the appropriate stop
    list is returned. 'english' is currently the only supported string
    value.
    There are several known issues with 'english' and you should
    consider an alternative (see :ref:`stop_words`).

    If a list, that list is assumed to contain stop words, all of which
    will be removed from the resulting tokens.
    Only applies if ``analyzer == 'word'``.

    If None, no stop words will be used. max_df can be set to a value
    in the range [0.7, 1.0) to automatically detect and filter stop
    words based on intra corpus document frequency of terms.

token_pattern : str
    Regular expression denoting what constitutes a "token", only used
    if ``analyzer == 'word'``. The default regexp selects tokens of 2
    or more alphanumeric characters (punctuation is completely ignored
    and always treated as a token separator).

ngram_range : tuple (min_n, max_n), default=(1, 1)
    The lower and upper boundary of the range of n-values for different
    n-grams to be extracted. All values of n such that min_n <= n <= max_n
    will be used. For example an ``ngram_range`` of ``(1, 1)`` means only
    unigrams, ``(1, 2)`` means unigrams and bigrams, and ``(2, 2)`` means
    only bigrams.
    Only applies if ``analyzer is not callable``.

max_df : float in range [0.0, 1.0] or int (default=1.0)
    When building the vocabulary ignore terms that have a document
    frequency strictly higher than the given threshold (corpus-specific
    stop words).
    If float, the parameter represents a proportion of documents, integer
    absolute counts.
    This parameter is ignored if vocabulary is not None.

min_df : float in range [0.0, 1.0] or int (default=1)
    When building the vocabulary ignore terms that have a document
    frequency strictly lower than the given threshold. This value is also
    called cut-off in the literature.
    If float, the parameter represents a proportion of documents, integer
    absolute counts.
    This parameter is ignored if vocabulary is not None.

max_features : int or None (default=None)
    If not None, build a vocabulary that only consider the top
    max_features ordered by term frequency across the corpus.

    This parameter is ignored if vocabulary is not None.

vocabulary : Mapping or iterable, optional (default=None)
    Either a Mapping (e.g., a dict) where keys are terms and values are
    indices in the feature matrix, or an iterable over terms. If not
    given, a vocabulary is determined from the input documents.

binary : bool (default=False)
    If True, all non-zero term counts are set to 1. This does not mean
    outputs will have only 0/1 values, only that the tf term in tf-idf
    is binary. (Set idf and normalization to False to get 0/1 outputs).

dtype : type, optional (default=float64)
    Type of the matrix returned by fit_transform() or transform().

norm : 'l1', 'l2' or None, optional (default='l2')
    Each output row will have unit norm, either:
    * 'l2': Sum of squares of vector elements is 1. The cosine
    similarity between two vectors is their dot product when l2 norm has
    been applied.
    * 'l1': Sum of absolute values of vector elements is 1.
    See :func:`preprocessing.normalize`.

use_idf : bool (default=True)
    Enable inverse-document-frequency reweighting.

smooth_idf : bool (default=True)
    Smooth idf weights by adding one to document frequencies, as if an
    extra document was seen containing every term in the collection
    exactly once. Prevents zero divisions.

sublinear_tf : bool (default=False)
    Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

Attributes
----------
vocabulary_ : dict
    A mapping of terms to feature indices.

fixed_vocabulary_: bool
    True if a fixed vocabulary of term to indices mapping
    is provided by the user

idf_ : array, shape (n_features)
    The inverse document frequency (IDF) vector; only defined
    if ``use_idf`` is True.

stop_words_ : set
    Terms that were ignored because they either:

      - occurred in too many documents (`max_df`)
      - occurred in too few documents (`min_df`)
      - were cut off by feature selection (`max_features`).

    This is only available if no vocabulary was given.

See Also
--------
CountVectorizer : Transforms text into a sparse matrix of n-gram counts.

TfidfTransformer : Performs the TF-IDF transformation from a provided
    matrix of counts.

Notes
-----
The ``stop_words_`` attribute can get large and increase the model size
when pickling. This attribute is provided only for introspection and can
be safely removed using delattr or set to None before pickling.

Examples
--------
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> corpus = [
...     'This is the first document.',
...     'This document is the second document.',
...     'And this is the third one.',
...     'Is this the first document?',
... ]
>>> vectorizer = TfidfVectorizer()
>>> X = vectorizer.fit_transform(corpus)
>>> print(vectorizer.get_feature_names())
['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
>>> print(X.shape)
(4, 9)
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `build_analyzer`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/feature_extraction/text.py#L415)
> `build_analyzer(self)`

Return a callable that handles preprocessing, tokenization
and n-grams generation.

Returns
-------
analyzer: callable
    A function to handle preprocessing, tokenization
    and n-grams generation.
### `build_preprocessor`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/feature_extraction/text.py#L305)
> `build_preprocessor(self)`

Return a function to preprocess the text before tokenization.

Returns
-------
preprocessor: callable
      A function to preprocess the text before tokenization.
### `build_tokenizer`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/feature_extraction/text.py#L333)
> `build_tokenizer(self)`

Return a function that splits a string into a sequence of tokens.

Returns
-------
tokenizer: callable
      A function to split a string into a sequence of tokens.
### `decode`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/feature_extraction/text.py#L192)
> `decode(self, doc)`

Decode the input into a string of unicode symbols.

The decoding strategy depends on the vectorizer parameters.

Parameters
----------
doc : str
    The string to decode.

Returns
-------
doc: str
    A string of unicode symbols.
### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/feature_extraction/text.py#L1819)
> `fit(self, raw_documents, y=None)`

Learn vocabulary and idf from training set.

Parameters
----------
raw_documents : iterable
    An iterable which yields either str, unicode or file objects.
y : None
    This parameter is not needed to compute tfidf.

Returns
-------
self : object
    Fitted vectorizer.
### `fit_transform`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/feature_extraction/text.py#L1840)
> `fit_transform(self, raw_documents, y=None)`

Learn vocabulary and idf, return term-document matrix.

This is equivalent to fit followed by transform, but more efficiently
implemented.

Parameters
----------
raw_documents : iterable
    An iterable which yields either str, unicode or file objects.
y : None
    This parameter is ignored.

Returns
-------
X : sparse matrix, [n_samples, n_features]
    Tf-idf-weighted document-term matrix.
### `get_feature_names`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/feature_extraction/text.py#L1306)
> `get_feature_names(self)`

Array mapping from feature integer indices to feature name.

Returns
-------
feature_names : list
    A list of feature names.
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
### `get_stop_words`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/feature_extraction/text.py#L346)
> `get_stop_words(self)`

Build or fetch the effective stop words list.

Returns
-------
stop_words: list or None
        A list of stop words.
### `inverse_transform`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/feature_extraction/text.py#L1275)
> `inverse_transform(self, X)`

Return terms per document with nonzero entries in X.

Parameters
----------
X : {array-like, sparse matrix} of shape (n_samples, n_features)
    Document-term matrix.

Returns
-------
X_inv : list of arrays, len = n_samples
    List of arrays of terms.
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L445)
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/feature_extraction/text.py#L1865)
> `transform(self, raw_documents, copy='deprecated')`

Transform documents to document-term matrix.

Uses the vocabulary and document frequencies (df) learned by fit (or
fit_transform).

Parameters
----------
raw_documents : iterable
    An iterable which yields either str, unicode or file objects.

copy : bool, default True
    Whether to copy X and operate on the copy or perform in-place
    operations.

    .. deprecated:: 0.22
       The `copy` parameter is unused and was deprecated in version
       0.22 and will be removed in 0.24. This parameter will be
       ignored.

Returns
-------
X : sparse matrix, [n_samples, n_features]
    Tf-idf-weighted document-term matrix.
