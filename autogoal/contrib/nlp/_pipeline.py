from sklearn.pipeline import Pipeline as _Pipeline

from . import _generated as gnlp
from . import _manual as mnlp
from autogoal.contrib.sklearn import _generated as sk
from autogoal.grammar import Union
from autogoal.contrib.sklearn._pipeline import Noop, SklearnClassifier

Tokenization = Union("Tokenization", gnlp.BlanklineTokenizer,\
                                     gnlp.LineTokenizer,\
                                     gnlp.MWETokenizer,\
                                     gnlp.SExprTokenizer,\
                                     gnlp.SpaceTokenizer,\
                                     gnlp.TabTokenizer,\
                                     gnlp.TextTilingTokenizer,\
                                     gnlp.ToktokTokenizer,\
                                     gnlp.TreebankWordTokenizer,\
                                     gnlp.TweetTokenizer,\
                                     gnlp.WhitespaceTokenizer,\
                                     gnlp.WordPunctTokenizer)

Stemmer = Union("Stemmer", gnlp.Cistem,\
                           gnlp.ISRIStemmer,\
                           gnlp.LancasterStemmer,\
                           gnlp.PorterStemmer,\
                           gnlp.RSLPStemmer,\
                           gnlp.SnowballStemmer)

Lemmatizer = Union("Lemmatizer", gnlp.WordNetLemmatizer)

Normalizer = Union("Normalizer", Stemmer, Lemmatizer, Noop)

Vectorization = Union("Vectorization", sk.TfidfVectorizer, sk.CountVectorizer, mnlp.Doc2Vec)


class TextPreprocessing(_Pipeline):
    def __init__(
        self, 
        tokenization:Tokenization,
        normalizing:Normalizer, 
        vectorization:Vectorization,
    ):
        self.tokenization = tokenization
        self.normalizing = normalizing
        self.vectorization = vectorization

        super().__init__(steps=[
            ('tokenizer', self.tokenization),
            ('normalizer', self.normalizing),
            ('vectorizer', self.vectorization),
        ])

class SklearnNLPClassifier(_Pipeline):
    def __init__(
        self,
        text_preprocessing: TextPreprocessing,
        classification: SklearnClassifier,
    ):
        self.text_preprocessing = text_preprocessing
        self.classification = classification
        
        super().__init__(steps = [
            ('text_preprocesser', self.text_preprocessing),
            ('classifier', self.classification)
        ])