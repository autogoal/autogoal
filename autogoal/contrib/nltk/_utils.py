from autogoal import kb
from autogoal.contrib.sklearn._utils import is_matrix_continuous_dense,\
                                            is_matrix_continuous_sparse,\
                                            is_categorical,\
                                            is_continuous,\
                                            is_string_list

DATA_RESOLVERS = {
    kb.MatrixContinuousDense(): is_matrix_continuous_dense,
    kb.MatrixContinuousSparse(): is_matrix_continuous_sparse,
    kb.CategoricalVector(): is_categorical,
    kb.ContinuousVector(): is_continuous,
    kb.List(kb.Word()): is_string_list,
    kb.List(kb.List(kb.Stem())): is_string_list_list,
}

DATA_TYPE_EXAMPLES = {
    kb.MatrixContinuousDense(): np.random.rand(10, 10),
    kb.MatrixContinuousSparse(): sp.rand(10, 10),
    kb.CategoricalVector(): np.asarray(["A"] * 5 + ["B"] * 5),
    kb.ContinuousVector(): np.random.rand(10),
    kb.DiscreteVector(): np.random.randint(0, 10, (10,), dtype=int),
    kb.List(kb.Document()): ["abc ipsu lorem say hello"] * 10,
    kb.List(kb.List(kb.Stem())): [["abc", "ipsu" "lorem"] * 10]
}

def is_algorithm(cls, verbose=False):
    if _is_classifier(cls):
        return "classifier"
    
    if _is_clusterer(cls):
        return "clusterer"
    
    if _is_sent_tokenizer(cls):
        return "sent_tokenizer"
    
    if _is_word_tokenizer(cls):
        return "word_tokenizer"
    
    if _is_lemmatizer(cls):
        return "lemmatizer"
    
    if _is_stemmer(cls):
        return "stemmer"
    
    if _is_word_embbeder(cls):
        return "word_embbeder"
    
    if _is_doc_embbeder(cls):
        return "doc_embbeder"

    return False

def _is_algorithm(cls, verbose = False):
    return  _is_stemmer(cls, verbose) or\
            _is_lemmatizer(cls, verbose) or\
            _is_word_tokenizer(cls, verbose) or\
            _is_sent_tokenizer(cls, verbose) or\
            _is_clusterer(cls, verbose) or\
            _is_classifier(cls, verbose) or\
            _is_word_embbeder(cls, verbose) or\
            _is_doc_embbeder(cls, verbose)

def _is_stemmer(cls, verbose=False):
    if hasattr(cls, "stem"):
        return True
    return False

def _is_lemmatizer(cls, verbose=False):
    if hasattr(cls, "lemmatize"):
        return True
    return False

def _is_word_tokenizer(cls, verbose=False):
    if not _is_sent_tokenizer(cls) and (hasattr(cls, "tokenize") or hasattr(cls, "word_tokenize")):
        return True
    return False

def _is_sent_tokenizer(cls, verbose=False):
    if "sentence" in str.lower(cls.__name__) or hasattr(cls, "sent_tokenize"):
            return True
    return False

def _is_clusterer(cls, verbose=False):
    if (hasattr(cls, "classify") and hasattr(cls, "cluster")):
        return True
    return False

def _is_classifier(cls, verbose = False):
    if (hasattr(cls, "classify") and hasattr(cls, "train")):
        return True
    return False

def _is_word_embbeder(cls, verbose = False):
    if (hasattr(cls, "build_vocab") and hasattr(cls, "train") and hasattr(cls, "wv")):
        return True
    return False

def _is_doc_embbeder(cls, verbose = False):
    if (hasattr(cls, "build_vocab") and hasattr(cls, "train") and hasattr(cls, "infer_vector")):
        return True
    return False


def is_stemmer(cls, verbose=False):
    """Determine if `cls` corresponds to something that resembles an nltk stemmer.
    If True, returns the valid (input, output) types.

    Examples:

    >>> from sklearn.linear_model import LogisticRegression, LinearRegression
    >>> is_classifier(LogisticRegression)
    (True, (MatrixContinuous(), CategoricalVector()))
    >>> is_classifier(LinearRegression)
    (False, None)

    """
    if not is_algorithm(cls, verbose=verbose):
        return False, None

    inputs = []

    for input_type in [kb.List(kb.Document())]:
        try:
            X = DATA_TYPE_EXAMPLES[input_type]
            y = DATA_TYPE_EXAMPLES[kb.List(kb.List(kb.Stem()))]

            stemmer = cls()
            y = [[stemmer.stem(word) for word in document] for document in X]

            assert is_string_list_list(y)
            inputs.append(input_type)
        except Exception as e:
            if verbose:
                warnings.warn(str(e))

    inputs = combine_types(*inputs)

    if inputs:
        return True, (inputs, kb.CategoricalVector())
    else:
        return False, None

def is_lemmatizer(cls, verbose=False):
    """Determine if `cls` corresponds to something that resembles an nltk lemmatizer.
    If True, returns the valid (input, output) types.

    Examples:

    >>> from sklearn.linear_model import LogisticRegression, LinearRegression
    >>> is_regressor(LogisticRegression)
    (False, None)
    >>> is_regressor(LinearRegression)
    (True, (MatrixContinuous(), ContinuousVector()))

    """
    if not is_algorithm(cls, verbose=verbose):
        return False, None

    inputs = []

    for input_type in [kb.List(kb.Document())]:
        try:
            X = DATA_TYPE_EXAMPLES[input_type]
            y = DATA_TYPE_EXAMPLES[kb.List(kb.List(kb.Stem()))]

            stemmer = cls()
            y = [[stemmer.lemmatize(word) for word in document] for document in X]

            assert is_string_list_list(y)
            inputs.append(input_type)
        except Exception as e:
            if verbose:
                warnings.warn(str(e))

    inputs = combine_types(*inputs)

    if inputs:
        return True, (inputs, kb.ContinuousVector())
    else:
        return False, None

def is_word_tokenizer(cls, verbose=False):
    """Determine if `cls` corresponds to something that resembles an sklearn clustering algorithm.
    If True, returns the valid (input, output) types.

    Examples:

    >>> from sklearn.linear_model import LogisticRegression, LinearRegression
    >>> is_clusterer(LogisticRegression)
    (False, None)
    >>> is_clusterer(LinearRegression)
    (False, None)
    >>> from sklearn.cluster import KMeans
    >>> is_clusterer(KMeans)
    (True, (MatrixContinuous(), DiscreteVector()))

    """
    if not is_algorithm(cls, verbose=verbose):
        return False, None

    inputs = []

    for input_type in [kb.MatrixContinuousDense(), kb.MatrixContinuousSparse()]:
        try:
            X = DATA_TYPE_EXAMPLES[input_type]

            clf = cls()
            y = clf.fit_predict(X)

            assert is_discrete(y)
            inputs.append(input_type)
        except Exception as e:
            if verbose:
                warnings.warn(str(e))

    inputs = combine_types(*inputs)

    if inputs:
        return True, (inputs, kb.DiscreteVector())
    else:
        return False, None

def is_sent_tokenizer(cls, verbose=False):
    """Determine if `cls` corresponds to something that resembles an sklearn general transformer.
    If True, returns the valid (input, output) types.

    Examples:

    >>> from sklearn.feature_extraction.text import CountVectorizer
    >>> is_transformer(CountVectorizer)
    (True, (List(Word()), MatrixContinuousSparse()))
    >>> from sklearn.decomposition.pca import PCA
    >>> is_transformer(PCA)
    (True, (MatrixContinuousDense(), MatrixContinuousDense()))

    """
    if not is_algorithm(cls, verbose=verbose):
        return False, None

    allowed_inputs = set()
    allowed_outputs = set()

    for input_type in [kb.MatrixContinuousDense(), kb.MatrixContinuousSparse(), kb.List(kb.Word())]:
        for output_type in [kb.MatrixContinuousDense(), kb.MatrixContinuousSparse(), kb.List(kb.Word())]:
            try:
                X = DATA_TYPE_EXAMPLES[input_type]

                clf = cls()
                X = clf.fit_transform(X)

                assert is_data_type(X, output_type)

                allowed_inputs.add(input_type)
                allowed_outputs.add(output_type)
            except Exception as e:
                if verbose:
                    warnings.warn(str(e))

    if len(allowed_outputs) != 1:
        return False, None

    inputs = combine_types(*allowed_inputs)

    if allowed_inputs:
        return True, (inputs, list(allowed_outputs)[0])
    else:
        return False, None

def is_clusterer(cls, verbose=False):
    """Determine if `cls` corresponds to something that resembles an sklearn classifier.
    If True, returns the valid (input, output) types.

    Examples:

    >>> from sklearn.linear_model import LogisticRegression, LinearRegression
    >>> is_classifier(LogisticRegression)
    (True, (MatrixContinuous(), CategoricalVector()))
    >>> is_classifier(LinearRegression)
    (False, None)

    """
    if not is_algorithm(cls, verbose=verbose):
        return False, None

    inputs = []

    for input_type in [kb.MatrixContinuousDense(), kb.MatrixContinuousSparse()]:
        try:
            X = DATA_TYPE_EXAMPLES[input_type]
            y = DATA_TYPE_EXAMPLES[kb.CategoricalVector()]

            clf = cls()
            clf.fit(X, y)
            y = clf.predict(X)

            assert is_categorical(y)
            inputs.append(input_type)
        except Exception as e:
            if verbose:
                warnings.warn(str(e))

    inputs = combine_types(*inputs)

    if inputs:
        return True, (inputs, kb.CategoricalVector())
    else:
        return False, None

def is_classifier(cls, verbose=False):
    """Determine if `cls` corresponds to something that resembles an sklearn regressor.
    If True, returns the valid (input, output) types.

    Examples:

    >>> from sklearn.linear_model import LogisticRegression, LinearRegression
    >>> is_regressor(LogisticRegression)
    (False, None)
    >>> is_regressor(LinearRegression)
    (True, (MatrixContinuous(), ContinuousVector()))

    """
    if not is_algorithm(cls, verbose=verbose):
        return False, None

    inputs = []

    for input_type in [kb.MatrixContinuousDense(), kb.MatrixContinuousSparse()]:
        try:
            X = DATA_TYPE_EXAMPLES[input_type]
            y = DATA_TYPE_EXAMPLES[kb.ContinuousVector()]

            clf = cls()
            clf.fit(X, y)
            y = clf.predict(X)

            assert is_continuous(y)
            inputs.append(input_type)
        except Exception as e:
            if verbose:
                warnings.warn(str(e))

    inputs = combine_types(*inputs)

    if inputs:
        return True, (inputs, kb.ContinuousVector())
    else:
        return False, None

def is_word_embbeder(cls, verbose=False):
    """Determine if `cls` corresponds to something that resembles an sklearn clustering algorithm.
    If True, returns the valid (input, output) types.

    Examples:

    >>> from sklearn.linear_model import LogisticRegression, LinearRegression
    >>> is_clusterer(LogisticRegression)
    (False, None)
    >>> is_clusterer(LinearRegression)
    (False, None)
    >>> from sklearn.cluster import KMeans
    >>> is_clusterer(KMeans)
    (True, (MatrixContinuous(), DiscreteVector()))

    """
    if not is_algorithm(cls, verbose=verbose):
        return False, None

    inputs = []

    for input_type in [kb.MatrixContinuousDense(), kb.MatrixContinuousSparse()]:
        try:
            X = DATA_TYPE_EXAMPLES[input_type]

            clf = cls()
            y = clf.fit_predict(X)

            assert is_discrete(y)
            inputs.append(input_type)
        except Exception as e:
            if verbose:
                warnings.warn(str(e))

    inputs = combine_types(*inputs)

    if inputs:
        return True, (inputs, kb.DiscreteVector())
    else:
        return False, None

def is_doc_embbeder(cls, verbose=False):
    """Determine if `cls` corresponds to something that resembles an sklearn general transformer.
    If True, returns the valid (input, output) types.

    Examples:

    >>> from sklearn.feature_extraction.text import CountVectorizer
    >>> is_transformer(CountVectorizer)
    (True, (List(Word()), MatrixContinuousSparse()))
    >>> from sklearn.decomposition.pca import PCA
    >>> is_transformer(PCA)
    (True, (MatrixContinuousDense(), MatrixContinuousDense()))

    """
    if not is_algorithm(cls, verbose=verbose):
        return False, None

    allowed_inputs = set()
    allowed_outputs = set()

    for input_type in [kb.MatrixContinuousDense(), kb.MatrixContinuousSparse(), kb.List(kb.Word())]:
        for output_type in [kb.MatrixContinuousDense(), kb.MatrixContinuousSparse(), kb.List(kb.Word())]:
            try:
                X = DATA_TYPE_EXAMPLES[input_type]

                clf = cls()
                X = clf.fit_transform(X)

                assert is_data_type(X, output_type)

                allowed_inputs.add(input_type)
                allowed_outputs.add(output_type)
            except Exception as e:
                if verbose:
                    warnings.warn(str(e))

    if len(allowed_outputs) != 1:
        return False, None

    inputs = combine_types(*allowed_inputs)

    if allowed_inputs:
        return True, (inputs, list(allowed_outputs)[0])
    else:
        return False, None


def is_data_type(X, data_type):
    return DATA_RESOLVERS[data_type](X)

IO_TYPE_HANDLER = [
    is_stemmer,
    is_lemmatizer,
    is_word_tokenizer,
    is_sent_tokenizer,
    is_clusterer,
    is_classifier,
    is_word_embbeder,
    is_doc_embbeder
]

def get_input_output(cls, verbose=False):
    for func in IO_TYPE_HANDLER:
        matches, types = func(cls, verbose=verbose)
        if matches:
            return types

    return None, None

def combine_types(*types):
    if len(types) == 1:
        return types[0]

    types = set(types)

    if types == {kb.MatrixContinuousDense(), kb.MatrixContinuousSparse()}:
        return kb.MatrixContinuous()

    return None

def is_string_list_list(obj):
    """Determines if `obj` is a sequence of sequence of strings.

    Examples:

    >>> is_string_list_list([['hello', world'], ['another'], ['sentence']])
    True
    >>> is_string_list_list(np.random.rand(10))
    False

    """
    try:
        oset = set()
        
        for sent in obj:
            for word in sent:
                oset.add(word)
                
        return len(oset) > 0.1 * len(obj) and all(isinstance(x, str) for x in oset)
    except:
        return False
    