# from sklearn.pipeline import Pipeline as _Pipeline

# from . import _generated as gnlp
# from . import _manual as mnlp
# from autogoal.contrib.sklearn import _generated as sk
# from autogoal.grammar import Union
# from autogoal.contrib.sklearn._pipeline import Noop, SklearnClassifier


# RawNormalizer = Union("RawNormalizer", mnlp.TextLowerer, Noop)

# Tokenization = Union("Tokenization", gnlp.MWETokenizer,\
#                                      gnlp.SpaceTokenizer,\
#                                      gnlp.ToktokTokenizer,\
#                                      gnlp.TreebankWordTokenizer,\
#                                      gnlp.TweetTokenizer,\
#                                      gnlp.WhitespaceTokenizer,\
#                                      gnlp.WordPunctTokenizer)

# Stopwords = Union("Stopwords", mnlp.StopwordRemover, Noop)

# Stemmer = Union("Stemmer", gnlp.Cistem,\
#                            gnlp.ISRIStemmer,\
#                            gnlp.LancasterStemmer,\
#                            gnlp.PorterStemmer,\
#                            gnlp.RSLPStemmer,\
#                            gnlp.SnowballStemmer)

# Lemmatizer = Union("Lemmatizer", gnlp.WordNetLemmatizer)

# Normalizer = Union("Normalizer", Stemmer, Lemmatizer, Noop)

# Vectorization = Union("Vectorization", sk.TfidfVectorizer, sk.CountVectorizer, mnlp.Doc2Vec)


# class TextPreprocessing(_Pipeline):
#     def __init__(
#         self, 
#         raw_normalizing:RawNormalizer,
#         tokenization:Tokenization,
#         stopwords_removing:Stopwords,
#         normalizing:Normalizer, 
#         vectorization:Vectorization,
#     ):
#         self.raw_normalizing = raw_normalizing
#         self.tokenization = tokenization
#         self.stopwords_removing = stopwords_removing
#         self.normalizing = normalizing
#         self.vectorization = vectorization

#         super().__init__(steps=[
#             ('raw_normalizer', self.raw_normalizing),
#             ('tokenizer', self.tokenization),
#             ('stopwords_remover', self.stopwords_removing),
#             ('normalizer', self.normalizing),
#             ('vectorizer', self.vectorization),
#         ])
 
# class SklearnNLPClassifier(_Pipeline):
#     def __init__(
#         self,
#         text_preprocessing: TextPreprocessing,
#         classification: SklearnClassifier,
#     ):
#         self.text_preprocessing = text_preprocessing
#         self.classification = classification
        
#         super().__init__(steps = [
#             ('text_preprocesser', self.text_preprocessing),
#             ('classifier', self.classification)
#         ])

# from _generated import PerceptronTagger, NEChunkParserTagger
# from _manual import GlobalChunker

# gc = GlobalChunker(PerceptronTagger(), NEChunkParserTagger())

# X = [[["Fernando", "Ruiz", "es", "un", "buen", "muchacho"]]]

# y = [[[("Erne", "B-NAME"), ("Estevanell", "I-NAME"), ("es","O"), ("el","O"), ("mejor","O")],
#      [("Jorgito", "B-NAME"), ("Estevanell", "I-NAME"), ("es","O"), ("muy","O"), ("bueno","O")],
#      [("el","O"), ("mecanico","O"), ("se","O"), ("llama","O"), ("Facundo", "B-NAME")],
#      [("Panfilo", "B-NAME"), ("es","O"), ("gracioso","O")]]]

# print(gc.run((X, y))[0])