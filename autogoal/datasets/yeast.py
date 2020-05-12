import numpy as np
import os
from autogoal.datasets import datapath, download
from sklearn.feature_extraction import DictVectorizer

def load():
    """
    Loads corpora from [Yeast uci dataset](https://archive.ics.uci.edu/ml/datasets/Yeast).

    ##### Examples

    ```python
    >>> X, y = load()
    >>> X.shape
    (1484, 8)
    >>> len(y)
    1484

    ```
    """

    try:
        download("yeast")
    except:
        print("Error loading data. This may be caused due to bad connection. Please delete badly downloaded data and retry")
        raise

    path = str(datapath(os.path.dirname(os.path.abspath(__file__)))) + "/data/yeast"
    f = open(os.path.join(path, "yeast.data"), "r")

    X = []
    y = []

    for i in f:
        clean_line = i.strip().split()
        temp = {}
        temp["1"] = float(clean_line[1])
        temp["2"] = float(clean_line[2])
        temp["3"] = float(clean_line[3])
        temp["4"] = float(clean_line[4])
        temp["5"] = float(clean_line[5])
        temp["6"] = float(clean_line[6])
        temp["7"] = float(clean_line[7])
        temp["8"] = float(clean_line[8])

        X.append(temp)
        y.append(clean_line[9])

    return _load_onehot(X, y)


def _load_onehot(X, y):
    vec = DictVectorizer(sparse=False)

    return vec.fit_transform(X), np.asarray(y)
