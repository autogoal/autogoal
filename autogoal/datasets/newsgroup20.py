# coding: utf-8

import os
import numpy as np
import math

from numpy.core.multiarray import concatenate
from autogoal.datasets import datapath, download_and_save, unpack, pad

URL_20_NEWS_GROUP = "http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz"

NEWS_CATEGORIES = [
    "alt.atheism",
    "comp.os.ms-windows.misc",
    "comp.sys.mac.hardware",
    "misc.forsale",
    "rec.motorcycles",
    "rec.sport.hockey",
    "sci.electronics",
    "sci.space",
    "talk.politics.guns",
    "talk.politics.misc",
    "comp.graphics",
    "comp.sys.ibm.pc.hardware",
    "comp.windows.x",
    "rec.autos",
    "rec.sport.baseball",
    "sci.crypt",
    "sci.med",
    "soc.religion.christian",
    "talk.politics.mideast",
    "talk.religion.misc",
]

TRAIN_FOLDER = "20news-bydate-train"
TEST_FOLDER = "20news-bydate-test"


def load(max_examples=None, padding_length=None, tokenizer=None):
    """
    Loads train and test datasets from 20newsgroup.

    ##### Examples

    ```python
    >>> X_train, y_train, X_valid, y_valid = load()
    >>> len(X_train), len(X_valid)
    (11314, 7532)
    >>> len(y_train), len(y_valid)
    (11314, 7532)

    ```
    """
    dataset = "20newsgroup"
    zip_fname = f"{dataset}.tar.gz"
    zip_path = datapath(zip_fname)
    news_path = datapath(dataset)
    try:
        if not zip_path.exists():
            url = URL_20_NEWS_GROUP

            download_and_save(url, zip_path, True)

            unpack(zip_fname, format="gztar", targetfile=dataset)
    except:
        print(
            "Error loading data. This may be caused due to bad connection. Please delete badly downloaded data and retry"
        )
        raise

    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for label, cat in enumerate(NEWS_CATEGORIES):
        folder = news_path / TRAIN_FOLDER / cat
        for file in os.listdir(folder):
            text = (folder / file).read_text(errors="ignore")
            if tokenizer and padding_length:
                words = pad(tokenizer.run(text), padding_length)
                text = " ".join(words)
            X_train.append(text)
            y_train.append(label)

        folder = news_path / TEST_FOLDER / cat
        for file in os.listdir(folder):
            text = (folder / file).read_text(errors="ignore")
            if tokenizer and padding_length:
                words = pad(tokenizer.run(text), padding_length)
                text = " ".join(words)
            X_test.append(text)
            y_test.append(label)

    if max_examples is not None:
        rng = np.random.default_rng(0)

        train_samples = math.ceil(max_examples * 0.6)
        train_data = list(zip(X_train, y_train))
        rng.shuffle(train_data)
        X_train, y_train = zip(*train_data[:train_samples])

        test_samples = math.floor(max_examples * 0.4)
        test_data = list(zip(X_test, y_test))
        rng.shuffle(test_data)
        X_test, y_test = zip(*test_data[:test_samples])

    return X_train, y_train, X_test, y_test


def label_to_category(*labels):
    categories = []
    for label in labels:
        category = NEWS_CATEGORIES[label]
        categories.append(category)
    return categories if len(categories) != 1 else categories[0]
