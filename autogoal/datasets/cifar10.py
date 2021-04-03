from autogoal.datasets import download, datapath
import pickle
import numpy as np


def load(training_batches=5):
    """
    Load the CIFAR-10 dataset

    ##### Parameters

    * 'training_batches': maximum number of batches to load for training, 
      each batch has 10,000 examples (min=`1`, max=`5`, default=`5`).

    ##### Examples

    >>> X_train, y_train, X_test, y_test = load(training_batches=5)
    >>> X_train.shape
    (50000, 32, 32, 3)
    >>> len(y_train)
    50000
    >>> X_test.shape
    (10000, 32, 32, 3)
    >>> len(y_test)
    10000
    >>> y_train[0]
    6

    """
    download("cifar10")

    X_train = []
    y_train = []

    for i in range(1, training_batches + 1):
        batch = datapath("cifar10") / f"data_batch_{i}"

        with open(batch, "rb") as fp:
            data = pickle.load(fp, encoding="bytes")
            X_train.append(data[b"data"])
            y_train.extend(data[b"labels"])

    X_train = np.vstack(X_train)
    X_train = np.reshape(X_train, (-1, 3, 32, 32)).transpose(0, 2, 3, 1)

    test_batch = datapath("cifar10") / "test_batch"

    with open(test_batch, "rb") as fp:
        data = pickle.load(fp, encoding="bytes")
        X_test, y_test = data[b"data"], data[b"labels"]
        X_test = np.reshape(X_test, (-1, 3, 32, 32)).transpose(0, 2, 3, 1)

    return X_train, y_train, X_test, y_test
