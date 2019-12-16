# coding: utf-8

import os
import numpy as np


def load_corpus():
    path = os.path.dirname(os.path.abspath(__file__))
    f = open(os.path.join(path, "german.data-numeric"), "r")

    X = []
    y = []

    for i in f.readlines():
        clean_line = i.strip().split()

        X.append([int(i) for i in clean_line[:-1]])
        y.append(int(clean_line[-1]))

    return np.asarray(X), np.asarray(y)
