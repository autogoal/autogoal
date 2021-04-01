import numpy as np
import os

from autogoal.datasets import download, datapath
from sklearn.feature_extraction import DictVectorizer


def load(representation="onehot"):
    download("uci_cars")

    f = open(datapath("uci_cars") / "car.data", "r")

    X = []
    y = []

    for i in f.readlines():
        clean_line = i.strip().split(",")

        temp = {}
        temp["buying"] = clean_line[0]
        temp["maint"] = clean_line[1]
        temp["doors"] = clean_line[2]
        temp["persons"] = clean_line[3]
        temp["lug_boot"] = clean_line[4]
        temp["safety"] = clean_line[5]

        X.append(temp)
        y.append(clean_line[6])

    if representation == "numeric":
        return _load_numeric(X, y)
    elif representation == "onehot":
        return _load_onehot(X, y)

    raise ValueError("Invalid value for represenation: %s" % representation)


def _load_numeric(X, y):
    new_X = []

    for d in X:
        new_d = {}
        for k, v in d.items():

            if k == "buying":
                if v == "vhigh":
                    new_d["buying"] = 4
                elif v == "high":
                    new_d["buying"] = 3
                elif v == "med":
                    new_d["buying"] = 2
                elif v == "low":
                    new_d["buying"] = 1

            if k == "maint":
                if v == "vhigh":
                    new_d["maint"] = 4
                elif v == "high":
                    new_d["maint"] = 3
                elif v == "med":
                    new_d["maint"] = 2
                elif v == "low":
                    new_d["maint"] = 1

            if k == "doors":
                if v == "5more":
                    new_d["doors"] = 4
                elif v == "4":
                    new_d["doors"] = 3
                elif v == "3":
                    new_d["doors"] = 2
                elif v == "2":
                    new_d["doors"] = 1

            if k == "persons":
                if v == "more":
                    new_d["persons"] = 3
                elif v == "4":
                    new_d["persons"] = 2
                elif v == "2":
                    new_d["persons"] = 1

            if k == "lug_boot":
                if v == "big":
                    new_d["lug_boot"] = 3
                elif v == "med":
                    new_d["lug_boot"] = 2
                elif v == "small":
                    new_d["lug_boot"] = 1

            if k == "safety":
                if v == "high":
                    new_d["safety"] = 3
                elif v == "med":
                    new_d["safety"] = 2
                elif v == "low":
                    new_d["safety"] = 1

        new_X.append(new_d)

    return _load_onehot(X, y)


def _load_onehot(X, y):
    vec = DictVectorizer(sparse=False)

    return vec.fit_transform(X), np.asarray(y)
