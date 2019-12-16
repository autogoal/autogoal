# coding: utf-8

import sys
import random
from hpopt.datasets.uci.car import load_corpus
from ..sklearn import SklearnClassifier, SklearnGrammar
from sklearn.model_selection import train_test_split


def main():
    X, y = load_corpus(representation='onehot')

    random.seed(0)

    for _ in range(20):
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3)
        classifier = SklearnClassifier(popsize=20, select=5, iters=100, timeout=300, global_timeout=3600, fitness_evaluations=5, verbose=True)
        classifier.fit(Xtrain, ytrain)

        with open("cars.log", "a") as fp:
            fp.write("%.5f, %.5f\n" % (classifier.score(Xtest, ytest), classifier.best_score_))


if __name__ == "__main__":
    main()
