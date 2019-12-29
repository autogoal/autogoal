# coding: utf8

from keras.layers import Input
from autogoal.ontology._keras import Bert


def main():
    x = Input((1000,))
    y = Bert()(x)


if __name__ == "__main__":
    main()
