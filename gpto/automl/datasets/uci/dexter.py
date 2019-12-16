# coding: utf-8

import os
import numpy as np
from scipy import sparse as sp


def load_corpus():
    path = os.path.dirname(os.path.abspath(__file__))
    train_data = open(os.path.join(path, "dexter_train.data"), "r")
    train_labels = open(os.path.join(path, "dexter_train.labels"), "r")
    valid_data = open(os.path.join(path, "dexter_valid.data"), "r")
    valid_labels = open(os.path.join(path, "dexter_valid.labels"), "r")

    Xtrain = sp.lil_matrix((300, 20000))
    ytrain = []
    Xvalid = sp.lil_matrix((300, 20000))
    yvalid = []

    for row, line in enumerate(train_data):
        for elem in line.split():
            col, value = elem.split(":")
            Xtrain[row, int(col)] = int(value)

    for row, line in enumerate(valid_data):
        for elem in line.split():
            col, value = elem.split(":")
            Xvalid[row, int(col)] = int(value)

    for line in train_labels:
        ytrain.append(int(line))

    for line in valid_labels:
        yvalid.append(int(line))

    return sp.vstack((Xtrain, Xvalid)).tocsr(), np.asarray(ytrain + yvalid)
