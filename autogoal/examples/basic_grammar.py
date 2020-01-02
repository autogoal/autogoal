# coding: utf8

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline as SkPipeline

from autogoal.grammar import (
    Continuous,
    Discrete,
    Categorical,
    Union,
    Boolean,
    generate_grammar,
)


class Count(CountVectorizer):
    def __init__(self, ngram: Discrete(1, 3)):
        super(Count, self).__init__(ngram_range=(1, ngram))


class TfIdf(TfidfVectorizer):
    def __init__(self, ngram: Discrete(1, 3), use_idf: Boolean()):
        super(TfIdf, self).__init__(ngram_range=(1, ngram), use_idf=use_idf)


Vectorizer = Union("Vectorizer", Count, TfIdf)


class NoDec:
    def fit_transform(self, X, y=None):
        return X

    def transform(self, X, y=None):
        return X

    def __repr__(self):
        return "NoDec()"


class SVD(TruncatedSVD):
    def __init__(self, n: Discrete(50, 200)):
        super(SVD, self).__init__(n_components=n)


Decomposer = Union("Decomposer", NoDec, SVD)


class LR(LogisticRegression):
    def __init__(self, penalty: Categorical("l1", "l2"), reg: Continuous(0, 10)):
        super(LR, self).__init__(penalty=penalty, C=reg)


class SVM(SVC):
    def __init__(self, kernel: Categorical("rbf", "linear", "poly"), reg: Continuous(0, 10)):
        super(SVM, self).__init__(C=reg, kernel=kernel)


class DT(DecisionTreeClassifier):
    def __init__(self, criterion: Categorical("gini", "entropy")):
        super(DT, self).__init__(criterion=criterion)


Classifier = Union("Classifier", LR, SVM, DT)


class Pipeline(SkPipeline):
    def __init__(
        self, vectorizer: Vectorizer, decomposer: Decomposer, classifier: Classifier
    ):
        self.vectorizer = vectorizer
        self.decomposer = decomposer
        self.classifier = classifier

        super(Pipeline, self).__init__(
            [
                ("vect", self.vectorizer),
                ("decomp", self.decomposer),
                ("class", self.classifier),
            ]
        )


def main():
    grammar = generate_grammar(Pipeline)
    print(grammar)

    instance = grammar.sample()
    print(instance)


if __name__ == "__main__":
    main()
