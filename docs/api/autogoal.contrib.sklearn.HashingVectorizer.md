# `autogoal.contrib.sklearn.HashingVectorizer`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L374)
> `HashingVectorizer(self, lowercase, n_features, binary, norm, alternate_sign)`

Convert a collection of text documents to a matrix of token occurrences

It turns a collection of text documents into a scipy.sparse matrix holding
token occurrence counts (or binary occurrence information), possibly
normalized as token frequencies if norm='l1' or projected on the euclidean
unit sphere if norm='l2'.

This text vectorizer implementation uses the hashing trick to find the
token string name to feature integer index mapping.

This strategy has several advantages:

- it is very low memory scalable to large datasets as there is no need to
  store a vocabulary dictionary in memory

- it is fast to pickle and un-pickle as it holds no state besides the
  constructor parameters

- it can be used in a streaming (partial fit) or parallel pipeline as there
  is no state computed during fit.

There are also a couple of cons (vs using a CountVectorizer with an
in-memory vocabulary):

- there is no way to compute the inverse transform (from feature indices to
  string feature names) which can be a problem when trying to introspect
  which features are most important to a model.

- there can be collisions: distinct tokens can be mapped to the same
  feature index. However in practice this is rarely an issue if n_features
  is large enough (e.g. 2 ** 18 for text classification problems).

- no IDF weighting as this would render the transformer stateful.

The hash function employed is the signed 32-bit version of Murmurhash3.

Read more in the :ref:`User Guide <text_feature_extraction>`.

Parameters
----------

input : string {'filename', 'file', 'content'}
    If 'filename', the sequence passed as an argument to fit is
    expected to be a list of filenames that need reading to fetch
    the raw content to analyze.

    If 'file', the sequence items must have a 'read' method (file-like
    object) that is called to fetch the bytes in memory.

    Otherwise the input is expected to be a sequence of items that
    can be of type string or byte.

encoding : string, default='utf-8'
    If bytes or files are given to analyze, this encoding is used to
    decode.

decode_error : {'strict', 'ignore', 'replace'}
    Instruction on what to do if a byte sequence is given to analyze that
    contains characters not of the given `encoding`. By default, it is
    'strict', meaning that a UnicodeDecodeError will be raised. Other
    values are 'ignore' and 'replace'.

strip_accents : {'ascii', 'unicode', None}
    Remove accents and perform other character normalization
    during the preprocessing step.
    'ascii' is a fast method that only works on characters that have
    an direct ASCII mapping.
    'unicode' is a slightly slower method that works on any characters.
    None (default) does nothing.

    Both 'ascii' and 'unicode' use NFKD normalization from
    :func:`unicodedata.normalize`.

lowercase : boolean, default=True
    Convert all characters to lowercase before tokenizing.

preprocessor : callable or None (default)
    Override the preprocessing (string transformation) stage while
    preserving the tokenizing and n-grams generation steps.
    Only applies if ``analyzer is not callable``.

tokenizer : callable or None (default)
    Override the string tokenization step while preserving the
    preprocessing and n-grams generation steps.
    Only applies if ``analyzer == 'word'``.

stop_words : string {'english'}, list, or None (default)
    If 'english', a built-in stop word list for English is used.
    There are several known issues with 'english' and you should
    consider an alternative (see :ref:`stop_words`).

    If a list, that list is assumed to contain stop words, all of which
    will be removed from the resulting tokens.
    Only applies if ``analyzer == 'word'``.

token_pattern : string
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

analyzer : string, {'word', 'char', 'char_wb'} or callable
    Whether the feature should be made of word or character n-grams.
    Option 'char_wb' creates character n-grams only from text inside
    word boundaries; n-grams at the edges of words are padded with space.

    If a callable is passed it is used to extract the sequence of features
    out of the raw, unprocessed input.

    .. versionchanged:: 0.21

    Since v0.21, if ``input`` is ``filename`` or ``file``, the data is
    first read from the file and then passed to the given callable
    analyzer.

n_features : integer, default=(2 ** 20)
    The number of features (columns) in the output matrices. Small numbers
    of features are likely to cause hash collisions, but large numbers
    will cause larger coefficient dimensions in linear learners.

binary : boolean, default=False.
    If True, all non zero counts are set to 1. This is useful for discrete
    probabilistic models that model binary events rather than integer
    counts.

norm : 'l1', 'l2' or None, optional
    Norm used to normalize term vectors. None for no normalization.

alternate_sign : boolean, optional, default True
    When True, an alternating sign is added to the features as to
    approximately conserve the inner product in the hashed space even for
    small n_features. This approach is similar to sparse random projection.

    .. versionadded:: 0.19

dtype : type, optional
    Type of the matrix returned by fit_transform() or transform().

Examples
--------
>>> from sklearn.feature_extraction.text import HashingVectorizer
>>> corpus = [
...     'This is the first document.',
...     'This document is the second document.',
...     'And this is the third one.',
...     'Is this the first document?',
... ]
>>> vectorizer = HashingVectorizer(n_features=2**4)
>>> X = vectorizer.fit_transform(corpus)
>>> print(X.shape)
(4, 16)

See Also
--------
CountVectorizer, TfidfVectorizer
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/feature_extraction/text.py#L741)
> `fit(self, X, y=None)`

Does nothing: this transformer is stateless.

Parameters
----------
X : array-like, shape [n_samples, n_features]
    Training data.
### `fit_transform`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/feature_extraction/text.py#L791)
> `fit_transform(self, X, y=None)`

Transform a sequence of documents to a document-term matrix.

Parameters
----------
X : iterable over raw text documents, length = n_samples
    Samples. Each sample must be a text document (either bytes or
    unicode strings, file name or file object depending on the
    constructor argument) which will be tokenized and hashed.
y : any
    Ignored. This parameter exists only for compatibility with
    sklearn.pipeline.Pipeline.

Returns
-------
X : sparse matrix of shape (n_samples, n_features)
    Document-term matrix.
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
### `partial_fit`

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/feature_extraction/text.py#L728)
> `partial_fit(self, X, y=None)`

Does nothing: this transformer is stateless.

This method is just there to mark the fact that this transformer
can work in a streaming setup.

Parameters
----------
X : array-like, shape [n_samples, n_features]
    Training data.
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_generated.py#L393)
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

> [📝](/usr/local/lib/python3.6/dist-packages/sklearn/feature_extraction/text.py#L761)
> `transform(self, X)`

Transform a sequence of documents to a document-term matrix.

Parameters
----------
X : iterable over raw text documents, length = n_samples
    Samples. Each sample must be a text document (either bytes or
    unicode strings, file name or file object depending on the
    constructor argument) which will be tokenized and hashed.

Returns
-------
X : sparse matrix of shape (n_samples, n_features)
    Document-term matrix.
