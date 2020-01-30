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
    kb.Postag():("lorem", "ipsum"), # (str, str) Tagged token
    kb.List(kb.Postag()):[("lorem", "ipsum")] * 10, # [(str, str), (str, str)] List of tagged tokens
    kb.List(kb.List(kb.Postag())):[[("lorem", "ipsum")] * 2], # [[(str, str), (str, str)], [(str, str), (str, str)]] List of Tagged Sentences
    kb.Chunktag():(("lorem", "ipsum"),"ipsum"), # ((str, str), str) IOB Tagged token
    kb.List(kb.Chunktag()):[(("lorem", "ipsum"),"ipsum")] * 10, # [((str, str), str), ((str, str), str)] List of IOB Tagged token
    kb.List(kb.List(kb.Chunktag())):[[(("lorem", "ipsum"),"ipsum")] * 2], # [[((str, str), str), ((str, str), str)], [((str, str), str), ((str, str), str)]] List of IOB Tagged Sentences
    kb.Stem():"ips",
    kb.Word():"ipsum",
    kb.Sentence():"It is the best of all movies.",
    kb.Document():"It is the best of all movies. I actually love that action scene.",
    kb.MatrixContinuousDense(): np.random.rand(10, 10),
    kb.MatrixContinuousSparse(): sp.rand(10, 10),
    kb.CategoricalVector(): np.asarray(["A"] * 5 + ["B"] * 5),
    kb.ContinuousVector(): np.random.rand(10),
    kb.DiscreteVector(): np.random.randint(0, 10, (10,), dtype=int),
    kb.List(kb.Word()):["ipsu", "lorem"],
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
    
    if is_pretrained_tagger(cls)[0]:
        return "trained_tagger"
    
    if _is_tagger(cls):
        return "tagger"
    
    return False

def _is_algorithm(cls, verbose = False):
    return  _is_stemmer(cls, verbose) or\
            _is_lemmatizer(cls, verbose) or\
            _is_word_tokenizer(cls, verbose) or\
            _is_sent_tokenizer(cls, verbose) or\
            _is_clusterer(cls, verbose) or\
            _is_classifier(cls, verbose) or\
            _is_word_embbeder(cls, verbose) or\
            _is_doc_embbeder(cls, verbose) or\
            _is_tagger(cls, verbose)

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

def _is_tagger(cls, verbose = False):
    if hasattr(cls, "tag"):
        return True
    return False

def _is_trained_tagger(cls, verbose = False):
    if hasattr(cls, "train") and _is_tagger(cls, verbose):
        return True
    return False

def _is_chunker(cls, verbose = False):
    if hasattr(cls, "parse"):
        return True
    return False
    

def is_stemmer(cls, verbose=False):
    """Determine if `cls` corresponds to something that resembles an nltk stemmer.
    If True, returns the valid (input, output) types.

    Examples:

    >>> from sklearn.linear_model import LogisticRegression
    >>> from nltk.stem import Cistem
    >>> is_stemmer(Cistem)
    (True, (Word(), Stem()))
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
    (True, (Word(), Stem()))
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
    >>> from nltk.tokenize import TweetTokenizer
    >>> is_word_tokenizer(TweetTokenizer)
    (True, (Sentence(), List(Word())))
    >>> is_word_tokenizer(LogisticRegression)
    (False, None)

    """
    if not _is_word_tokenizer(cls, verbose=verbose):
        return False, None

    inputs = []
    output = kb.List(kb.Word())

    for input_type in [kb.Sentence()]:
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
    (True, (Document(), List(Sentence())))
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

def is_tagger(cls, verbose=False):
    """Determine if `cls` corresponds to something that resembles an nltk pos tagger.
    If True, returns the valid (input, output) types.

    Examples:

    >>> from nltk.tag import AffixTagger
    >>> from nltk.tokenize import PunktSentenceTokenizer
    >>> is_tagger(AffixTagger)
    (True, (List(List(Word())), List(List(Postag()))))
    >>> is_tagger(LogisticRegression)
    (False, None)

    """
    if not _is_tagger(cls, verbose=verbose):
        return False, None

    inputs = []
    output = kb.List(kb.List(kb.Postag()))

    for input_type in [kb.List(kb.List(kb.Word()))]:
        try:
            X = DATA_TYPE_EXAMPLES[input_type]

            X_train = [[(word,word) for word in sentence] for sentence in X]
            
            tagger = cls(train=X_train)
            # tagger = cls()
            y = tagger.tag_sents(X)

            assert DATA_RESOLVERS[output](y)
            inputs.append(input_type)
        except Exception as e:
            if verbose:
                warnings.warn(str(e))

    inputs = combine_types(*inputs)

    if inputs:
        return True, (inputs, output)
    else:
        is_ptt = is_pretrained_tagger(cls, verbose)
        is_ckr = is_chunker(cls, verbose)
        
        if is_ptt[0]:
            return is_ptt
        if is_ckr[0]:
            return is_ckr
        return False, None

def is_chunker(cls, verbose=False):
    """Determine if `cls` corresponds to something that resembles an nltk chunker.
    If True, returns the valid (input, output) types.

    Examples:

    >>> from nltk.chunk.named_entity import NEChunkParserTagger
    >>> from nltk.tokenize import PunktSentenceTokenizer
    >>> is_chunker(NEChunkParserTagger)
    (True, (List(List(Tuple(Word(), Word()))), List(List(Tuple(Tuple(Word(), Word()), Word())))))
    >>> is_chunker(PunktSentenceTokenizer)
    (False, None)

    """
    if not _is_tagger(cls, verbose=verbose):
        return False, None

    inputs = []
    output = kb.List(kb.List(kb.Chunktag()))

    for input_type in [kb.List(kb.List(kb.Postag()))]:
        try:
            X = DATA_TYPE_EXAMPLES[input_type]

            X_train = [[((word, postag), postag) for word, postag in sentence] for sentence in X]
            
            chunker = cls(train=X_train)
            y = chunker.tag_sents(X)

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

def is_pretrained_tagger(cls, verbose=False):
    """Determine if `cls` corresponds to something that resembles an nltk sentence tokenizer.
    If True, returns the valid (input, output) types.

    Examples:

    >>> from nltk.tag import AffixTagger
    >>> from nltk.tag.perceptron import PerceptronTagger
    >>> is_pretrained_tagger(PerceptronTagger)
    (True, (List(Word()), List(Tuple(Word(), Word()))))
    >>> is_pretrained_tagger(AffixTagger)
    (False, None)

    """
    if not _is_tagger(cls, verbose=verbose):
        return False, None

    inputs = []
    output = kb.List(kb.Postag())

    for input_type in [kb.List(kb.Word())]:
        try:
            X = DATA_TYPE_EXAMPLES[input_type]
            
            tagger = cls()
            
            y = tagger.tag(X)

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
    
def is_data_type(X, data_type):
    return DATA_RESOLVERS[data_type](X)

IO_TYPE_HANDLER = [
    is_stemmer,
    is_lemmatizer,
    is_word_tokenizer,
    is_sent_tokenizer,
    is_clusterer,
    is_classifier,
    is_tagger,
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

    >>> is_word_list(['hello', 'world'])
    True
    >>> is_word_list(np.random.rand(10))
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

    >>> is_word_list_list([['hello'], ['another'], ['word']])
    True
    >>> is_word_list_list(np.random.rand(10))
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

    >>> is_word('hello')
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

    >>> is_sentence('hello world')
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

    >>> is_text_list_list([['hello', 'world'], ['another'], ['sentence']])
    True
    >>> is_text_list_list(np.random.rand(10))
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

def is_tag(obj):
    """Determines if `obj` is a tuple of two strings.

    Examples:

    >>> is_tag(('hello', 'yes'))
    True
    >>> is_tag(('hi', 22))
    False

    """
    try:
        return isinstance(obj, tuple) and len(obj) == 2 and all((isinstance(x, str) or x == None) for x in obj)
    except:
        return False
    
def is_tag_list(obj):
    """Determines if `obj` is a list of tuple of two strings.

    Examples:

    >>> is_tag_list([('hello', 'yes'), ('how', 'is')])
    True
    >>> is_tag_list(('hello', 'yes'))
    False

    """
    try:
        return isinstance(obj, list) and all(is_tag(x) for x in obj)
    except:
        return False

def is_tagged_sentence_list(obj):
    """Determines if `obj` is a list(list(tuple(str, str)))

    Examples:

    >>> is_tagged_sentence_list([[('hello', 'yes'), ('how', 'is')]])
    True
    >>> is_tagged_sentence_list([('hello', 'yes')])
    False

    """
    try:
        return isinstance(obj, list) and all(is_tag_list(x) for x in obj)
    except:
        return False


def is_chunk(obj):
    """Determines if `obj` is a tuple(tuple(str, str), str).

    Examples:

    >>> is_chunk((('hello', 'yes'),'how'))
    True
    >>> is_chunk(('hello', 'yes'))
    False

    """
    try:
        return isinstance(obj, tuple) and len(obj) == 2 and is_tag(obj[0]) and isinstance(obj[1], str)
    except:
        return False

def is_chunk_list(obj):
    """Determines if `obj` is a list(tuple(tuple(str, str), str)).

    Examples:

    >>> is_chunk_list([(('hello', 'yes'),'how'), (('are', 'you'), 'today')])
    True
    >>> is_chunk_list([('hello', 'yes'), ('how', 'are')])
    False

    """
    try:
        return isinstance(obj, list) and all(is_chunk(x) for x in obj)
    except:
        return False

def is_chunked_sentence_list(obj):
    """Determines if `obj` is a list(list(tuple(tuple(str, str), str))).

    Examples:

    >>> is_chunked_sentence_list([[(('hello', 'yes'),'how'), (('are', 'you'), 'today')]])
    True
    >>> is_chunked_sentence_list([('hello', 'yes'), ('how', 'are')])
    False

    """
    try:
        return isinstance(obj, list) and all(is_chunk_list(x) for x in obj)
    except:
        return False


DATA_RESOLVERS = {
    kb.Postag():is_tag,
    kb.List(kb.Postag()):is_tag_list,
    kb.List(kb.List(kb.Postag())):is_tagged_sentence_list,
    kb.Chunktag:is_chunk,
    kb.List(kb.Chunktag()):is_chunk_list,
    kb.List(kb.List(kb.Chunktag())):is_chunked_sentence_list,
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