# coding: utf-8

import os
import numpy as np
from scipy import sparse as sp
from autogoal.datasets import datapath, download

def load_corpus():
    """
    Loads train and valid datasets from [DOROTHEA uci dataset](https://archive.ics.uci.edu/ml/datasets/dorothea).

    ##### Examples

    ```python
    >>> X_train, X_valid, y_train, y_valid = load_corpus()
    >>> X_train.shape, X_valid.shape
    (800, 100000) (350, 100000)
    >>> len(y_train), len(y_valid)
    800 350
    ```
    """
    
    try:
        download("dorothea")
    except:
        print("Error loading data. This may be caused due to bad connection. Please delete badly downloaded data and retry")
        raise
    
    path = str(datapath(os.path.dirname(os.path.abspath(__file__)))) + "/data/dorothea"
    train_data = open(os.path.join(path, "dorothea_train.data"), "r")
    train_labels = open(os.path.join(path, "dorothea_train.labels"), "r")
    valid_data = open(os.path.join(path, "dorothea_valid.data"), "r")
    valid_labels = open(os.path.join(path, "dorothea_valid.labels"), "r")

    Xtrain = sp.lil_matrix((800, 100000), dtype=int)
    ytrain = []
    Xvalid = sp.lil_matrix((350, 100000), dtype=int)
    yvalid = []

    for row, line in enumerate(train_data):
        for col in line.split():
            Xtrain[row, int(col)-1] = True

    for row, line in enumerate(valid_data):
        for col in line.split():
            Xvalid[row, int(col)-1] = False

    for line in train_labels:
        ytrain.append(int(line))

    for line in valid_labels:
        yvalid.append(int(line))

    return Xtrain, Xvalid, ytrain, yvalid
