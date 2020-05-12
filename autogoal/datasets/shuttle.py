import numpy as np
import os
from autogoal.datasets import datapath, download
from sklearn.feature_extraction import DictVectorizer


def load(max_examples=None):
    """
    Loads train and valid datasets from [Shuttle uci dataset](https://archive.ics.uci.edu/ml/datasets/Statlog+(Shuttle)).

    ##### Examples

    ```python
    >>> X_train, y_train, X_valid, y_valid = load()
    >>> X_train.shape, X_valid.shape
    ((43500, 9), (14500, 9))
    >>> len(y_train), len(y_valid)
    (43500, 14500)

    ```
    """

    try:
        download("shuttle")
    except:
        print("Error loading data. This may be caused due to bad connection. Please delete badly downloaded data and retry")
        raise

    path = str(datapath(os.path.dirname(os.path.abspath(__file__)))) + "/data/shuttle"
    train_data = open(os.path.join(path, "shuttle.trn"), "r")
    test_data = open(os.path.join(path, "shuttle.tst"), "r")

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    for i in train_data.readlines():
        clean_line = i.strip().split()

        temp = {}
        temp["1"] = int(clean_line[0])
        temp["2"] = int(clean_line[1])
        temp["3"] = int(clean_line[2])
        temp["4"] = int(clean_line[3])
        temp["5"] = int(clean_line[4])
        temp["6"] = int(clean_line[5])
        temp["7"] = int(clean_line[6])
        temp["8"] = int(clean_line[7])
        temp["9"] = int(clean_line[8])

        X_train.append(temp)
        y_train.append(clean_line[9])

        if max_examples and len(X_train) >= max_examples:
            break

    for i in test_data.readlines():
        clean_line = i.strip().split()

        temp = {}
        temp["1"] = int(clean_line[0])
        temp["2"] = int(clean_line[1])
        temp["3"] = int(clean_line[2])
        temp["4"] = int(clean_line[3])
        temp["5"] = int(clean_line[4])
        temp["6"] = int(clean_line[5])
        temp["7"] = int(clean_line[6])
        temp["8"] = int(clean_line[7])
        temp["9"] = int(clean_line[8])

        X_test.append(temp)
        y_test.append(clean_line[9])

        if max_examples and len(X_test) >= max_examples:
            break

    X_train, y_train = _load_onehot(X_train, y_train)
    X_test, y_test = _load_onehot(X_test, y_test)

    return X_train, y_train, X_test, y_test


def _load_onehot(X, y):
    vec = DictVectorizer(sparse=False)

    return vec.fit_transform(X), np.asarray(y)
