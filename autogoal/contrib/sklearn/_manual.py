from sklearn.feature_extraction.text import CountVectorizer as _CountVectorizer
from autogoal.contrib.sklearn._builder import SklearnTransformer
from autogoal.kb import List, Sentence, MatrixContinuousSparse, algorithm, Word
from autogoal.grammar import Boolean


class CountVectorizerNoTokenize(_CountVectorizer, SklearnTransformer):
    def __init__(
        self,
        lowercase: Boolean(),
        binary: Boolean(),
        inner_tokenizer: algorithm(Sentence(), List(Word())),
    ):
        self.inner_tokenizer = inner_tokenizer

        SklearnTransformer.__init__(self)
        _CountVectorizer.__init__(self, lowercase=lowercase, binary=binary)

    def build_tokenizer(self):
        def func(sentence):
            return self.inner_tokenizer.run(sentence)

        return func

    def run(self, input: List(Sentence())) -> MatrixContinuousSparse():
        return SklearnTransformer.run(self, input)

