# coding: utf-8

import random
from hpopt.datasets.uci.dexter import load_corpus
from ..sklearn import SklearnClassifier


def main():
    X, y = load_corpus()

    random.seed(0)
    classifier = SklearnClassifier(popsize=100, select=20, iters=10, timeout=10, verbose=True)
    classifier.fit(X, y)


if __name__ == "__main__":
    main()
