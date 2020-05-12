# coding: utf-8

import os
import numpy as np
from scipy import sparse as sp
from autogoal.datasets import datapath, download

def load():
    """
    Loads train and valid datasets from [Gisette uci dataset](https://archive.ics.uci.edu/ml/datasets/Gisette).

    ##### Examples

    ```python
    >>> X_train, y_train, X_valid, y_valid = load()
    >>> X_train.shape, X_valid.shape
    ((6000, 5000), (1000, 5000))
    >>> len(y_train), len(y_valid)
    (6000, 1000)

    ```
    """

    try:
        download("gisette")
    except:
        print("Error loading data. This may be caused due to bad connection. Please delete badly downloaded data and retry")
        raise

    path = str(datapath(os.path.dirname(os.path.abspath(__file__)))) + "/data/gisette"

    train_data = open(os.path.join(path, "gisette_train.data"), "r")
    train_labels = open(os.path.join(path, "gisette_train.labels"), "r")
    valid_data = open(os.path.join(path, "gisette_valid.data"), "r")
    valid_labels = open(os.path.join(path, "gisette_valid.labels"), "r")

    Xtrain = sp.lil_matrix((6000, 5000))
    ytrain = []
    Xvalid = sp.lil_matrix((1000, 5000))
    yvalid = []

    for i, line in enumerate(train_data):
        for j, value in enumerate(line.split()):
            value = int(value)
            if value > 0:
                Xtrain[i,j] = value

    for i, line in enumerate(valid_data):
        for j, value in enumerate(line.split()):
            value = int(value)
            if value > 0:
                Xvalid[i,j] = value

    for line in train_labels:
        ytrain.append(int(line) > 0)

    for line in valid_labels:
        yvalid.append(int(line) > 0)

    return Xtrain.tocsr(), np.asarray(ytrain), Xvalid.tocsr(), np.asarray(yvalid)
