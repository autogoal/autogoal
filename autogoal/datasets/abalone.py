import numpy as np
import os
from sklearn.feature_extraction import DictVectorizer
from autogoal.datasets import datapath, download


def load(representation='numeric'):
    """
    Loads corpora from [ABALONE uci dataset](https://archive.ics.uci.edu/ml/datasets/Abalone).

    ##### Examples

    ```python
    >>> X, y = load()
    >>> X.shape
    (4177, 6047)
    >>> len(y)
    4177

    ```
    """

    try:
        download("abalone")
    except:
        print("Error loading data. This may be caused due to bad connection. Please delete badly downloaded data and retry")
        raise

    path = str(datapath(os.path.dirname(os.path.abspath(__file__)))) + "/data/abalone"
    f = open(os.path.join(path, "abalone.data"), "r")

    X = []
    y = []

    for i in f.readlines():
        clean_line = i.strip().split(",")

        temp = {}
        temp["Sex"] = clean_line[0]
        temp["Length"] = clean_line[1]
        temp["Diameter"] = clean_line[2]
        temp["Height"] = clean_line[3]
        temp["Shucked weight"] = clean_line[4]
        temp["Whole weight"] = clean_line[5]
        temp["Viscera weight"] = clean_line[6]
        temp["Shell weight"] = clean_line[7]

        X.append(temp)
        y.append(clean_line[8])

    if representation == 'numeric':
        return _load_numeric(X, y)
    elif representation == 'onehot':
        return _load_onehot(X, y)

    raise ValueError("Invalid value for represenation: %s" % representation)


def _load_numeric(X, y):
    new_X = []

    for d in X:
        new_d = d.copy()

        v = d['Sex']
        if v == "M":
            new_d["Sex"] = 3
        elif v == "F":
            new_d["Sex"] = 2
        elif v == "I":
            new_d["Sex"] = 1

        new_X.append(new_d)

    return _load_onehot(new_X, y)


def _load_onehot(X, y):
    vec = DictVectorizer(sparse=False)

    return vec.fit_transform(X), np.asarray(y)
