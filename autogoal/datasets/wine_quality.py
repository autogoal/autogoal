# coding: utf-8

import numpy as np
import os

from autogoal.datasets import download, datapath


def load(white=True, red=True, max_examples=None):
    if not red and not white:
        raise ValueError("Either red or white must be selected")

    download("wine_quality")

    f_white = open(datapath("wine_quality") / "winequality-white.csv", "r")
    f_red = open(datapath("wine_quality") / "winequality-red.csv", "r")

    X = []
    y = []

    if white:
        title_line = True
        for i in f_white.readlines():

            if max_examples and len(X) >= max_examples:
                break

            if title_line == True:
                title_line = False
                continue

            clean_line = i.strip().split(";")

            X.append([1, 0] + [float(i) for i in clean_line[:-1]])
            y.append(float(clean_line[-1]))

    if red:
        title_line = True
        for i in f_red.readlines():

            if max_examples and len(X) >= max_examples:
                break

            if title_line == True:
                title_line = False
                continue

            clean_line = i.strip().split(";")

            X.append([0, 1] + [float(i) for i in clean_line[:-1]])
            y.append(float(clean_line[-1]))

    return np.asarray(X), np.asarray(y)
