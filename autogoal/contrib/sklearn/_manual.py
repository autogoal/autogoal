from sklearn.feature_extraction.text import CountVectorizer as _CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn_crfsuite import CRF

from autogoal.contrib.sklearn._builder import SklearnTransformer, SklearnEstimator
from autogoal.kb import *
from autogoal.grammar import (
    BooleanValue,
    CategoricalValue,
    DiscreteValue,
    ContinuousValue,
)
from autogoal.utils import nice_repr
from autogoal.kb import AlgorithmBase, Supervised


@nice_repr
class CountVectorizerNoTokenize(_CountVectorizer, SklearnTransformer):
    def __init__(
        self,
        lowercase: BooleanValue(),
        stopwords_remove: BooleanValue(),
        binary: BooleanValue(),
        inner_tokenizer: algorithm(Sentence, Seq[Word]),
        inner_stemmer: algorithm(Word, Stem),
        inner_stopwords: algorithm(Seq[Word], Seq[Word]),
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

    def run(self, input: Seq[Sentence]) -> MatrixContinuousSparse:
        return SklearnTransformer.run(self, input)


class _FeatureVectorizer(SklearnTransformer):
    def __init__(self, sparse):
        self.vectorizer = DictVectorizer(sparse=sparse)
        super().__init__()

    def fit_transform(self, X, y=None):
        return self.vectorizer.fit_transform(X)

    def transform(self, X, y=None):
        return self.vectorizer.transform(X, y=y)


@nice_repr
class FeatureSparseVectorizer(_FeatureVectorizer):
    def __init__(self):
        super().__init__(sparse=True)

    def run(self, input: Seq[FeatureSet]) -> MatrixContinuousSparse:
        return super().run(input)


@nice_repr
class FeatureDenseVectorizer(_FeatureVectorizer):
    def __init__(self):
        super().__init__(sparse=False)

    def run(self, input: Seq[FeatureSet]) -> MatrixContinuousDense:
        return super().run(input)


@nice_repr
class CRFTagger(CRF, SklearnEstimator):
    def __init__(
        self, algorithm: CategoricalValue("lbfgs", "l2sgd", "ap", "pa", "arow")
    ) -> None:
        SklearnEstimator.__init__(self)
        super().__init__(algorithm=algorithm)

    def run(
        self, X: Seq[Seq[FeatureSet]], y: Supervised[Seq[Seq[Label]]]
    ) -> Seq[Seq[Label]]:
        return SklearnEstimator.run(self, X, y)


__all__ = [
    "CountVectorizerNoTokenize",
    "FeatureSparseVectorizer",
    "FeatureDenseVectorizer",
    "CRFTagger",
]
