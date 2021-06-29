# coding: utf-8

import os
import random
import math

from numpy.core.multiarray import concatenate
from autogoal.datasets import datapath, download_and_save, unpack, pad, shuffle_lists

URL_IMDB = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
REVIEW_POLARITY = ["positive", "negative"]


def load(max_examples=None, padding_length=None, tokenizer=None):
    """
    Loads train and test datasets from 20newsgroup.

    ##### Examples

    ```python
    >>> X_train, y_train, X_valid, y_valid = load()
    >>> len(X_train), len(X_valid)
    (25000, 25000)
    >>> len(y_train), len(y_valid)
    (25000, 25000)

    ```
    """
    dataset = "imdb"
    zip_fname = f"{dataset}.tar.gz"
    zip_path = datapath(zip_fname)
    pos_train_path = datapath(dataset) / "aclImdb" / "train" / "pos"
    neg_train_path = datapath(dataset) / "aclImdb" / "train" / "neg"
    pos_test_path = datapath(dataset) / "aclImdb" / "test" / "pos"
    neg_test_path = datapath(dataset) / "aclImdb" / "test" / "neg"
    try:
        if not zip_path.exists():
            url = URL_IMDB

            download_and_save(url, zip_path, True)

            unpack(zip_fname, format="gztar", targetfile=dataset)
    except:
        print(
            "Error loading data. This may be caused due to bad connection. Please delete badly downloaded data and retry"
        )
        raise

    folder_samples = math.floor(max_examples / 4) if max_examples is not None else None
    rng = random.Random(0)

    X_train_pos, y_train_pos = __load_folder(
        pos_train_path, tokenizer, padding_length, 0, folder_samples
    )
    X_train_neg, y_train_neg = __load_folder(
        neg_train_path, tokenizer, padding_length, 1, folder_samples
    )

    X_train, y_train = shuffle_lists(
        rng, X_train_pos + X_train_neg, y_train_pos + y_train_neg
    )

    X_test_pos, y_test_pos = __load_folder(
        pos_test_path, tokenizer, padding_length, 0, folder_samples
    )
    X_test_neg, y_test_neg = __load_folder(
        neg_test_path, tokenizer, padding_length, 1, folder_samples
    )

    X_test, y_test = shuffle_lists(
        rng, X_test_pos + X_test_neg, y_test_pos + y_test_neg
    )

    return X_train, y_train, X_test, y_test


def __load_folder(folder, tokenizer, padding_length, label, max_examples):
    X = []
    y = []
    for i, file in enumerate(os.listdir(folder)):
        if max_examples is not None and i >= max_examples:
            break
        text = (folder / file).read_text(errors="ignore")
        if tokenizer and padding_length:
            words = pad(tokenizer.run(text), padding_length)
            text = " ".join(words)
        X.append(text)
        y.append(label)
    return X, y


def label_to_category(*labels):
    categories = []
    for label in labels:
        category = REVIEW_POLARITY[label]
        categories.append(category)
    return categories if len(categories) != 1 else categories[0]
