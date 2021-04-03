# coding: utf-8

import os
import numpy as np
from scipy import sparse as sp
from autogoal.datasets import datapath, download


def load():
    """
    Loads train and valid datasets from [DOROTHEA uci dataset](https://archive.ics.uci.edu/ml/datasets/dorothea).

    ##### Examples

    ```python
    >>> X_train, y_train, X_valid, y_valid = load()
    >>> X_train.shape, X_valid.shape
    ((800, 100000), (350, 100000))
    >>> len(y_train), len(y_valid)
    (800, 350)

    ```
    """

    download("dorothea")

    train_data = open(datapath("dorothea") / "dorothea_train.data", "r")
    train_labels = open(datapath("dorothea") / "dorothea_train.labels", "r")
    valid_data = open(datapath("dorothea") / "dorothea_valid.data", "r")
    valid_labels = open(datapath("dorothea") / "dorothea_valid.labels", "r")

    Xtrain = sp.lil_matrix((800, 100000), dtype=int)
    ytrain = []
    Xvalid = sp.lil_matrix((350, 100000), dtype=int)
    yvalid = []

    for row, line in enumerate(train_data):
        for col in line.split():
            Xtrain[row, int(col) - 1] = 1

    for row, line in enumerate(valid_data):
        for col in line.split():
            Xvalid[row, int(col) - 1] = 1

    for line in train_labels:
        ytrain.append(int(line))

    for line in valid_labels:
        yvalid.append(int(line))

    return Xtrain.tocsr(), np.asarray(ytrain), Xvalid.tocsr(), np.asarray(yvalid)
