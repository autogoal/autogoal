"""
This module generates a random dataset useful for quickly testing the interface of AutoGOAL methods.
"""

import numpy as np


def load(samples=100, classes=2, features=10, seed=None):
    """
    Create a random X,y pair.

    ##### Examples

    ```python
    >>> X, y = load(samples=4, features=2, seed=0)
    >>> print(X)
    [[0.5488135  0.71518937]
     [0.60276338 0.54488318]
     [0.4236548  0.64589411]
     [0.43758721 0.891773  ]]
    >>> y
    array([0, 0, 0, 1])

    ```
    """

    if seed is not None:
        np.random.seed(seed)

    X = np.random.random((samples, features))
    y = np.random.randint(0, classes, samples).astype(str)

    return X, y