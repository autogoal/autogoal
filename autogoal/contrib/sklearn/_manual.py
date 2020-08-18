from sklearn.feature_extraction.text import CountVectorizer as _CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn_crfsuite import CRF

from autogoal.contrib.sklearn._builder import SklearnTransformer, SklearnEstimator
from autogoal.kb import (
    List,
    Sentence,
    MatrixContinuousSparse,
    MatrixContinuousDense,
    algorithm,
    Word,
    Stem,
    Flags,
    Vector,
    Category,
    Tuple,
)
from autogoal.grammar import Boolean, Categorical, Discrete, Continuous
from autogoal.utils import nice_repr


@nice_repr
class CountVectorizerNoTokenize(_CountVectorizer, SklearnTransformer):
    def __init__(
        self,
        lowercase: Boolean(),
        stopwords_remove: Boolean(),
        binary: Boolean(),
        inner_tokenizer: algorithm(Sentence(), List(Word())),
        inner_stemmer: algorithm(Word(), Stem()),
        inner_stopwords: algorithm(List(Word()), List(Word())),
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
            tokens = (
                self.inner_stopwords.run(sentence) if self.stopwords_remove else tokens
            )
            return [self.inner_stemmer.run(token) for token in tokens]

        return func

    def run(self, input: List(Sentence())) -> MatrixContinuousSparse():
        return SklearnTransformer.run(self, input)


class _FlagsVectorizer(SklearnTransformer):
    def __init__(self, sparse):
        self.vectorizer = DictVectorizer(sparse=sparse)
        super().__init__()

    def fit_transform(self, X, y=None):
        return self.vectorizer.fit_transform(X)

    def transform(self, X, y=None):
        return self.vectorizer.transform(X, y=y)

@nice_repr
class FlagsSparseVectorizer(_FlagsVectorizer):
    def __init__(self):
        super().__init__(sparse=True)

    def run(self, input: List(Flags())) -> MatrixContinuousSparse():
        return super().run(input)


@nice_repr
class FlagsDenseVectorizer(_FlagsVectorizer):
    def __init__(self):
        super().__init__(sparse=False)

    def run(self, input: List(Flags())) -> MatrixContinuousDense():
        return super().run(input)


@nice_repr
class CRFTagger(CRF, SklearnEstimator):
    def __init__(self, algorithm: Categorical('lbfgs', 'l2sgd', 'ap', 'pa', 'arow')) -> None:
        SklearnEstimator.__init__(self)
        super().__init__(algorithm=algorithm)

    def run(self, input: Tuple(List(List(Flags())), List(List(Category())))) -> List(List(Category())):
        return SklearnEstimator.run(self, input)


__all__ = [
    "CountVectorizerNoTokenize",
    "FlagsSparseVectorizer",
    "FlagsDenseVectorizer",
    "CRFTagger"
]