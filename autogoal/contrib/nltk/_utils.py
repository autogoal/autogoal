import warnings
import inspect
import re
import numpy as np
import scipy.sparse as sp

from autogoal import kb
from autogoal.contrib.sklearn._utils import is_matrix_continuous_dense,\
                                            is_matrix_continuous_sparse,\
                                            is_categorical,\
                                            is_continuous,\
                                            is_string_list

DATA_TYPE_EXAMPLES = {
    kb.Stem():"ips",
    kb.Word():"ipsum",
    kb.Sentence():"It is the best of all movies.",
    kb.Document():"It is the best of all movies. I actually love that action scene.",
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
    if (hasattr(cls, "tokenize") or hasattr(cls, "word_tokenize")):
        return True
    return False

def _is_sent_tokenizer(cls, verbose=False):
    if hasattr(cls, "sent_tokenize") or (hasattr(cls, "tokenize")):
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
    output = kb.Stem()

    for input_type in [kb.Word()]:
        try:
            X = DATA_TYPE_EXAMPLES[input_type]

            stemmer = cls()
            y = stemmer.stem(X)

            assert DATA_RESOLVERS[output](y)
            inputs.append(input_type)
        except Exception as e:
            if verbose:
                warnings.warn(str(e))

    inputs = combine_types(*inputs)

    if inputs:
        return True, (inputs, output)
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
    output = kb.Stem()
    
    for input_type in [kb.Word()]:
        try:
            X = DATA_TYPE_EXAMPLES[input_type]

            lemmatizer = cls()
            y = lemmatizer.lemmatize(X)

            assert DATA_RESOLVERS[output](y)
            inputs.append(input_type)
        except Exception as e:
            if verbose:
                warnings.warn(str(e))

    inputs = combine_types(*inputs)

    if inputs:
        return True, (inputs, output)
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
    output = kb.List(kb.Word())

    for input_type in [kb.Document()]:
        try:
            X = DATA_TYPE_EXAMPLES[input_type]

            tokenizer = cls()
            y = tokenizer.tokenize(X)

            assert DATA_RESOLVERS[output](y)
            inputs.append(input_type)
        except Exception as e:
            if verbose:
                warnings.warn(str(e))

    inputs = combine_types(*inputs)

    if inputs:
        return True, (inputs, output)
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
    output = kb.List(kb.Sentence())

    for input_type in [kb.Document()]:
        try:
            X = DATA_TYPE_EXAMPLES[input_type]

            tokenizer = cls()
            y = tokenizer.tokenize(X)

            assert DATA_RESOLVERS[output](y)
            inputs.append(input_type)
        except Exception as e:
            if verbose:
                warnings.warn(str(e))

    inputs = combine_types(*inputs)

    if inputs:
        return True, (inputs, output)
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

def is_word(obj):
    """Determines if `obj` is a sequence of sequence of strings.

    Examples:

    >>> is_word('hello'])
    True
    >>> is_word(np.random.rand(10))
    False

    """
    try:        
        return isinstance(obj, str) and len(obj.split()) == 1
    except:
        return False

def is_sentence(obj):
    """Determines if `obj` is a sentence strings.

    Examples:

    >>> is_word('hello'])
    True
    >>> is_word(np.random.rand(10))
    False

    """
    try:        
        return isinstance(obj, str) and len(obj.split()) > 1
    except:
        return False

def is_sentence_list(obj):
    try:
        return all([is_sentence(x) for x in obj])
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
    kb.Stem():is_word,
    kb.Word():is_word,
    kb.Sentence():is_sentence,
    kb.Document():is_sentence,
    kb.MatrixContinuousDense(): is_matrix_continuous_dense,
    kb.MatrixContinuousSparse(): is_matrix_continuous_sparse,
    kb.CategoricalVector(): is_categorical,
    kb.ContinuousVector(): is_continuous,
    kb.List(kb.Word()): is_word_list,
    kb.List(kb.Sentence()): is_sentence_list,
    kb.List(kb.List(kb.Stem())): is_word_list_list,
    kb.List(kb.List(kb.Word())): is_word_list_list,
}

def find_classes(include=".*", exclude=None):
    """
    Returns the list of all `nltk` wrappers in `autogoal`.

    You can pass filters to include or exclude specific classes.
    The filters are regular expressions that are matched against
    the names of the classes. Only classes that pass the `include` filter
    and not the `exclude` filter will be returned.
    By default all classes are returned.

    ##### Parameters

    - `include`: regular expression to match for including classes. Defaults to `".*"`, i.e., all classes.
    - `exclude`: regular expression to match for excluding classes. Defaults to `None`.

    ##### Examples

    ```python
    >>> from pprint import pprint
    >>> pprint(find_classes(include='.*Classifier', exclude='.*Tree.*'))
    [<class 'autogoal.contrib.sklearn._generated.KNeighborsClassifier'>,
     <class 'autogoal.contrib.sklearn._generated.PassiveAggressiveClassifier'>,
     <class 'autogoal.contrib.sklearn._generated.RidgeClassifier'>,
     <class 'autogoal.contrib.sklearn._generated.SGDClassifier'>]

    ```
    """
    import autogoal.contrib.nltk._generated as module
    from autogoal.contrib.nltk._builder import NltkClassifier, NltkClusterer, NltkLemmatizer, NltkStemmer, NltkTokenizer

    return [
        c for n, c in inspect.getmembers(
            module,
            lambda c: inspect.isclass(c)
            and issubclass(c, (NltkClusterer, NltkClassifier, NltkLemmatizer, NltkStemmer, NltkTokenizer))
            and re.match(include, c.__name__)
            and (exclude is None or not re.match(exclude, c.__name__)),
        )
    ]