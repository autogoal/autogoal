from sklearn.pipeline import Pipeline as _Pipeline

from . import _generated as nlp
from autogoal.contrib.sklearn import _generated as sk
from autogoal.grammar import Union
from autogoal.contrib.sklearn._pipeline import Noop, SklearnClassifier


Tokenization = Union("Tokenization", nlp.BlanklineTokenizer, nlp.LineTokenizer, nlp.MWETokenizer,\
                    nlp.SExprTokenizer, nlp.SpaceTokenizer, nlp.TabTokenizer, nlp.TextTilingTokenizer,\
                    nlp.ToktokTokenizer, nlp.TreebankWordTokenizer, nlp.TweetTokenizer, nlp.WhitespaceTokenizer, nlp.WordPunctTokenizer)
Stemmer = Union("Stemmer", nlp.Cistem, nlp.ISRIStemmer, nlp.LancasterStemmer, nlp.PorterStemmer,\
                nlp.RSLPStemmer, nlp.SnowballStemmer, Noop)
Vectorization = Union("Vectorization", sk.TfidfVectorizer, sk.CountVectorizer)


class TextPreprocessing(_Pipeline):
    def __init__(
        self, 
        tokenization:Tokenization,
        stemming:Stemmer, 
        vectorization:Vectorization,
    ):
        self.tokenization = tokenization
        self.stemming = stemming
        self.vectorization = vectorization

        super().__init__(steps=[
            ('tokenizer', self.tokenization),
            ('stemmer', self.stemming),
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