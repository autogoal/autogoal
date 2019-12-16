# coding: utf8

from ..hmlopt.datasets.uci.car import load_corpus
from ..automl import AutoML
from ..ontology import onto

from sklearn.model_selection import train_test_split


def main():
    X, y =load_corpus(representation='onehot')

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.7)

    automl = AutoML(errors='warn', verbose=True)
    automl.optimize(Xtrain, ytrain)

    print(automl.score(Xtest, ytest))


if __name__ == "__main__":
    main()

