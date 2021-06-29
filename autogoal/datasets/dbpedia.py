# coding: utf-8

import os
import random
import math
import csv

from numpy.core.multiarray import concatenate
from autogoal.datasets import datapath, download_and_save, shuffle_lists, unpack, pad

# BUG Not actually working, would need drive dependency
URL_DBPEDIA = (
    "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k"
)
DATA_CATEGORIES = [
    "Company",
    "EducationalInstitution",
    "Artist",
    "Athlete",
    "OfficeHolder",
    "MeanOfTransportation",
    "Building",
    "NaturalPlace",
    "Village",
    "Animal",
    "Plant",
    "Album",
    "Film",
    "WrittenWork",
]


def load(max_examples=None, padding_length=None, tokenizer=None):
    """
    Loads train and test datasets from 20newsgroup.

    ##### Examples

    ```python
    >>> X_train, y_train, X_valid, y_valid = load()
    >>> len(X_train), len(X_valid)
    (560000, 70000)
    >>> len(y_train), len(y_valid)
    (560000, 70000)

    ```
    """
    dataset = "dbpedia"
    zip_fname = f"{dataset}.tar.gz"
    zip_path = datapath(zip_fname)
    train_file = datapath(dataset) / "dbpedia_csv" / "train.csv"
    test_file = datapath(dataset) / "dbpedia_csv" / "test.csv"
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    try:
        if not zip_path.exists():
            url = URL_DBPEDIA

            download_and_save(url, zip_path, True)
            unpack(zip_fname, format="gztar", targetfile=dataset)

    except:
        print(
            "Error loading data. This may be caused due to bad connection. Please delete badly downloaded data and retry"
        )
        raise

    rng = random.Random(0)

    for label, title, comment in csv_line_reader(train_file):
        text = title + "\n" + comment
        if tokenizer and padding_length:
            words = pad(tokenizer.run(text), padding_length)
            text = " ".join(words)
        X_train.append(text)
        y_train.append(label)

    print("train read")

    for label, title, comment in csv_line_reader(test_file):
        text = title + "\n" + comment
        if tokenizer and padding_length:
            words = pad(tokenizer.run(text), padding_length)
            text = " ".join(words)
        X_test.append(text)
        y_test.append(label)

    X_train, y_train = shuffle_lists(rng, X_train, y_train)
    X_test, y_test = shuffle_lists(rng, X_test, y_test)

    if max_examples is not None:
        train_samples = math.floor(max_examples * 0.89)
        test_samples = math.ceil(max_examples * 0.11)
        X_train = X_train[:train_samples]
        y_train = y_train[:train_samples]
        X_test = X_test[:test_samples]
        y_test = y_test[:test_samples]

    return X_train, y_train, X_test, y_test


def csv_line_reader(file_name):
    with open(file_name, "r") as file:
        csv_reader = csv.reader(file, delimiter=",", quotechar='"')
        for _, line in enumerate(csv_reader):
            label = int(line[0]) - 1
            title = line[1]
            comment = line[2]
            yield (label, title, comment)


def label_to_category(*labels):
    categories = []
    for label in labels:
        category = DATA_CATEGORIES[label]
        categories.append(category)
    return categories if len(categories) != 1 else categories[0]
