from sklearn.feature_extraction.text import CountVectorizer as _CountVectorizer
from autogoal.contrib.sklearn._builder import SklearnTransformer
from autogoal.kb import List, Sentence, MatrixContinuousSparse, algorithm, Word, Stem
from autogoal.grammar import Boolean


class CountVectorizerNoTokenize(_CountVectorizer, SklearnTransformer):
    def __init__(
        self,
        lowercase: Boolean(),
        stopwords_remove:Boolean(),
        binary: Boolean(),
        inner_tokenizer: algorithm(Sentence(), List(Word())),
        inner_stemmer: algorithm(Word(), Stem()),
        inner_stopwords: algorithm(List(Word()), List(Word()))
    ):
        self.stopwords_remove = stopwords_remove
        self.inner_tokenizer = inner_tokenizer
        self.inner_stemmer = inner_stemmer
        self.inner_stopwords = inner_stopwords

        SklearnTransformer.__init__(self)
        _CountVectorizer.__init__(self, lowercase=lowercase, binary=binary)

    def build_tokenizer(self):
        def func(sentence):
            tokens = self.inner_tokenizer.run(sentence)
            tokens = self.inner_stopwords.run(sentence) if self.stopwords_remove else tokens
            return [self.inner_stemmer.run(token) for token in tokens]

        return func

    def run(self, input: List(Sentence())) -> MatrixContinuousSparse():
        return SklearnTransformer.run(self, input)

