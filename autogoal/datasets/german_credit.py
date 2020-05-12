# coding: utf-8

import os
import numpy as np


from autogoal.datasets import download, datapath
from sklearn.feature_extraction import DictVectorizer


def _parse(x):
    return int(x) if x.isdigit() else x

def load(max_examples=None):
    download("german_credit")

    f = open(datapath("german_credit") / "german.data", "r")

    X = []
    y = []

    for i in f.readlines():

        if max_examples and len(X) >= max_examples:
            break

        clean_line = i.strip().split()

        line = {'feature_%i'% i : _parse(v) for i,v in enumerate(clean_line[:-1])}

        X.append(line)
        y.append(int(clean_line[-1]) == 2)

    return DictVectorizer(sparse=False).fit_transform(X), np.asarray(y)
