from autogoal.datasets import download, datapath

import numpy as np
import pandas as pd


def load(unrolled=True):
    """
    Load the MNIST dataset.

    ##### Parameters

    * 'unrolled': Wether to return unrolled images 

    ##### Examples

    Loading unrolled images:

    ```python
    >>> X_train, y_train, X_test, y_test = load()
    >>> X_train.shape
    (60000, 784)
    >>> len(y_train)
    60000
    >>> X_test.shape
    (10000, 784)
    >>> len(y_test)
    10000
    >>> y_train[0]
    5

    ```

    Loading full images:

    ```python
    >>> X_train, y_train, X_test, y_test = load(unrolled=False)
    >>> X_train.shape
    (60000, 28, 28, 1)
    >>> len(y_train)
    60000
    >>> X_test.shape
    (10000, 28, 28, 1)
    >>> len(y_test)
    10000
    >>> y_train[0]
    5

    ```
    """
    download("mnist")

    X_train_data = pd.read_csv(datapath('mnist') / 'mnist_train.csv')
    X_test_data = pd.read_csv(datapath('mnist') / 'mnist_test.csv')
    
    X_train = [] 
    y_train = []
    X_test = []
    y_test = []

    for i in range(len(X_train_data)):
        if unrolled:
            X_train.append(X_train_data.iloc[i,1:])
        else:
            X_train.append(X_train_data.iloc[i,1:].values.reshape((28,28,1)))
        y_train.append(X_train_data.iloc[i,0])

    for i in range(len(X_test_data)):
        if unrolled:
            X_test.append(X_test_data.iloc[i,1:])
        else:
            X_test.append(X_test_data.iloc[i,1:].values.reshape((28,28,1)))
        y_test.append(X_test_data.iloc[i,0])

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
