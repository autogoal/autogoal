import warnings

import numpy as np
import scipy.sparse as sp

from autogoal import kb
from autogoal.contrib.sklearn._utils import is_matrix_continuous_dense,\
                                            is_matrix_continuous_sparse,\
                                            is_categorical,\
                                            is_continuous,\
                                            is_string_list

DATA_TYPE_EXAMPLES = {
    kb.MatrixContinuousDense(): np.random.rand(10, 10),
    kb.MatrixContinuousSparse(): sp.rand(10, 10),
    kb.CategoricalVector(): np.asarray(["A"] * 5 + ["B"] * 5),
    kb.ContinuousVector(): np.random.rand(10),
    kb.DiscreteVector(): np.random.randint(0, 10, (10,), dtype=int),
    kb.List(kb.Document()): ["abc ipsu lorem say hello", "ipsum lorem", "abc"] * 2,
    kb.List(kb.List(kb.Stem())): [["abc", "ipsu", "lorem"] * 10],
    kb.List(kb.List(kb.Word())): [["abc", "ipsu", "lorem"] * 10],
    kb.List(kb.List(kb.Sentence())): [["abc a sentence lorem"], ["ipsum lorem"], ["abc"]]
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
    if ("sentence" in str.lower(cls.__name__) or hasattr(cls, "sent_tokenize")) and (hasattr(cls, "tokenize")):
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

    >>> from sklearn.linear_model import LogisticRegression
    >>> from nltk.stem import Cistem
    >>> is_stemmer(Cistem)
    (True, (List(List(Word())), List(List(Stem())))
    >>> is_stemmer(LogisticRegression)
    (False, None)

    """
    if not _is_stemmer(cls, verbose=verbose):
        return False, None

    inputs = []

    for input_type in [kb.List(kb.List(kb.Word()))]:
        try:
            X = DATA_TYPE_EXAMPLES[input_type]

            stemmer = cls()
            y = [[stemmer.stem(word) for word in document] for document in X]

            assert is_word_list_list(y)
            inputs.append(input_type)
        except Exception as e:
            if verbose:
                warnings.warn(str(e))

    inputs = combine_types(*inputs)

    if inputs:
        return True, (inputs, kb.List(kb.List(kb.Stem())))
    else:
        return False, None

def is_lemmatizer(cls, verbose=False):
    """Determine if `cls` corresponds to something that resembles an nltk lemmatizer.
    If True, returns the valid (input, output) types.

    Examples:

    >>> from sklearn.linear_model import LogisticRegression
    >>> from nltk.stem import WordNetLemmatizer
    >>> is_lemmatizer(WordNetLemmatizer)
    (True, (List(List(Word())), List(List(Stem())))
    >>> is_lemmatizer(LogisticRegression)
    (False, None)

    """
    if not _is_lemmatizer(cls, verbose=verbose):
        return False, None

    inputs = []

    for input_type in [kb.List(kb.List(kb.Word()))]:
        try:
            X = DATA_TYPE_EXAMPLES[input_type]

            stemmer = cls()
            y = [[stemmer.lemmatize(word) for word in document] for document in X]

            assert is_word_list_list(y)
            inputs.append(input_type)
        except Exception as e:
            if verbose:
                warnings.warn(str(e))

    inputs = combine_types(*inputs)

    if inputs:
        return True, (inputs, kb.List(kb.List(kb.Stem())))
    else:
        return False, None

def is_word_tokenizer(cls, verbose=False):
    """Determine if `cls` corresponds to something that resembles an nltk word tokenizer algorithm.
    If True, returns the valid (input, output) types.

    Examples:

    >>> from sklearn.linear_model import LogisticRegression
    >>> from nltk.tokenize import PunktWordTokenizer
    >>> is_word_tokenizer(PunktWordTokenizer)
    (True, (List(Document()), List(List(Word())))
    >>> is_word_tokenizer(LogisticRegression)
    (False, None)

    """
    if not _is_word_tokenizer(cls, verbose=verbose):
        return False, None

    inputs = []

    for input_type in [kb.List(kb.Document())]:
        try:
            X = DATA_TYPE_EXAMPLES[input_type]

            tokenizer = cls()
            y = [tokenizer.tokenize(x) for x in X]

            assert is_word_list_list(y)
            inputs.append(input_type)
        except Exception as e:
            if verbose:
                warnings.warn(str(e))

    inputs = combine_types(*inputs)

    if inputs:
        return True, (inputs, kb.List(kb.List(kb.Word())))
    else:
        return False, None

def is_sent_tokenizer(cls, verbose=False):
    """Determine if `cls` corresponds to something that resembles an nltk sentence tokenizer.
    If True, returns the valid (input, output) types.

    Examples:

    >>> from sklearn.linear_model import LogisticRegression
    >>> from nltk.tokenize import PunktSentenceTokenizer
    >>> is_sent_tokenizer(PunktSentenceTokenizer)
    (True, (List(Document()), List(List(Word())))
    >>> is_sent_tokenizer(LogisticRegression)
    (False, None)

    """
    if not _is_sent_tokenizer(cls, verbose=verbose):
        return False, None

    inputs = []

    for input_type in [kb.List(kb.Document())]:
        try:
            X = DATA_TYPE_EXAMPLES[input_type]

            tokenizer = cls()
            y = [tokenizer.tokenize(x) for x in X]

            assert is_text_list_list(y)
            inputs.append(input_type)
        except Exception as e:
            if verbose:
                warnings.warn(str(e))

    inputs = combine_types(*inputs)

    if inputs:
        return True, (inputs, kb.List(kb.List(kb.Word())))
    else:
        return False, None

def is_clusterer(cls, verbose=False):
    """Determine if `cls` corresponds to something that resembles an nltk clusterer.
    If True, returns the valid (input, output) types.

    Examples:

    >>> from sklearn.linear_model import LogisticRegression
    >>> from nltk.cluster import GAAClusterer
    >>> is_clusterer(GAAClusterer)
    (True, (MatrixContinuousDense(), CategoricalVector()))
    >>> is_clusterer(LogisticRegression)
    (False, None)

    """
    if not _is_clusterer(cls, verbose=verbose):
        return False, None

    inputs = []

    for input_type in [kb.MatrixContinuousDense(), kb.MatrixContinuousSparse()]:
        try:
            X = DATA_TYPE_EXAMPLES[input_type]
            y = DATA_TYPE_EXAMPLES[kb.CategoricalVector()]

            clusterer = cls()
            clusterer.cluster(X)
            y = [clusterer.classify(x) for x in X]

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
    """Determine if `cls` corresponds to something that resembles an nltk classifier.
    If True, returns the valid (input, output) types.
    """
    if not _is_algorithm(cls, verbose=verbose):
        return False, None

    inputs = []

    #TODO: Fix somehow compatibility with nltk classifiers
    
    inputs = combine_types(*inputs)

    if inputs:
        return True, (inputs, kb.ContinuousVector())
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
    # is_word_embbeder,
    # is_doc_embbeder
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

def is_word_list(obj):
    """Determines if `obj` is a sequence of sequence of strings.

    Examples:

    >>> is_string_list_list([['hello', world'], ['another'], ['sentence']])
    True
    >>> is_string_list_list(np.random.rand(10))
    False

    """
    try:
        oset = set()
        
        for word in obj:
            if len(word.split()) > 1:
                return False
            oset.add(word)
                
        return len(oset) > 0.1 * len(obj) and all(isinstance(x, str) for x in oset)
    except:
        return False

def is_word_list_list(obj):
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
                if len(word.split()) > 1:
                    return False
                oset.add(word)
                
        return len(oset) > 0.1 * len(obj) and all(isinstance(x, str) for x in oset)
    except:
        return False

def is_text_list_list(obj):
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
            for text in sent:
                oset.add(text)
                
        return len(oset) > 0.1 * len(obj) and all(isinstance(x, str) for x in oset)
    except:
        return False

    
DATA_RESOLVERS = {
    kb.MatrixContinuousDense(): is_matrix_continuous_dense,
    kb.MatrixContinuousSparse(): is_matrix_continuous_sparse,
    kb.CategoricalVector(): is_categorical,
    kb.ContinuousVector(): is_continuous,
    kb.List(kb.Word()): is_word_list,
    kb.List(kb.List(kb.Stem())): is_word_list_list,
    kb.List(kb.List(kb.Word())): is_word_list_list,
}