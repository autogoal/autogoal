# coding: utf-8

import os
import numpy as np


from autogoal.datasets import download, datapath

def load(max_examples=None):
    download("german_credit")

    f = open(datapath("german_credit") / "german.data-numeric", "r")

    X = []
    y = []

    for i in f.readlines():

        if max_examples and len(X) >= max_examples:
            break

        clean_line = i.strip().split()

        X.append([int(i) for i in clean_line[:-1]])
        y.append(int(clean_line[-1]))


    return np.asarray(X), np.asarray(y)
