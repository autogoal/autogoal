from autogoal.kb import algorithm, Sentence, List, Word, ContinuousVector, MatrixContinuousDense
from autogoal.grammar import Categorical
from autogoal.utils import nice_repr

import numpy as np


__all__ = ["SentenceVectorizer", "AggregateMerger", "DocumentVectorizer"]


@nice_repr
class SentenceVectorizer:
    def __init__(
        self,
        tokenizer: algorithm(Sentence(), List(Word())),
        vectorizer: algorithm(Word(), ContinuousVector()),
        merger: algorithm(List(ContinuousVector()), ContinuousVector()),
    ):
        self.tokenizer = tokenizer
        self.vectorizer = vectorizer
        self.merger = merger

    def run(self, input: Sentence()) -> ContinuousVector():
        tokens = self.tokenizer.run(input)
        vectors = [self.vectorizer.run(token) for token in tokens]
        return self.merger.run(vectors)


@nice_repr
class DocumentVectorizer:
    def __init__(self, vectorizer: SentenceVectorizer):
        self.vectorizer = vectorizer

    def run(self, input: List(Sentence())) -> MatrixContinuousDense():
        return np.vstack([self.vectorizer.run(s) for s in input])


@nice_repr
class AggregateMerger:
    def __init__(self, mode: Categorical("mean", "max")):
        self.mode = mode

    def run(self, input: List(ContinuousVector())) -> ContinuousVector():
        input = np.asarray(input)

        if self.mode == "mean":
            return input.mean(axis=1)
        elif self.mode == "max":
            return input.max(axis=1)

        raise ValueError("Invalid mode: %s" % self.mode)
