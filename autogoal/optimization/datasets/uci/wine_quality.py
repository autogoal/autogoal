# coding: utf-8

import numpy as np
import os


def load_corpus(white=False, red=False):
    if not red and not white:
        raise ValueError("Either red or white must be selected")

    path = os.path.dirname(os.path.abspath(__file__))

    f_white = open(os.path.join(path, "winequality-white.csv"), "r")
    f_red = open(os.path.join(path, "winequality-red.csv"), "r")

    X = []
    y = []

    if white:
        title_line = True
        for i in f_white.readlines():

            if title_line == True:
                title_line = False
                continue

            clean_line = i.strip().split(";")

            X.append([float(i) for i in clean_line[:-1]])
            y.append(float(clean_line[-1]))

    if red:
        title_line = True
        for i in f_red.readlines():

            if title_line == True:
                title_line = False
                continue

            clean_line = i.strip().split(";")

            X.append([float(i) for i in clean_line[:-1]])
            y.append(float(clean_line[-1]))

    return np.asarray(X), np.asarray(y)
