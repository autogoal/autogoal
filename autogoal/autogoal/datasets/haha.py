import os
import pandas as pd
import numpy as np

from autogoal.datasets import datapath, download


def load_raw(max_examples=None):
    """
    Loads the train and test datasets for the [HAHA 2019 corpus](https://www.fing.edu.uy/inco/grupos/pln/haha/index.html#data)
    as Pandas dataframes.

    ##### Examples

    ```python
    >>> train, test = load_raw()
    >>> len(train), len(test)
    (24000, 6000)
    >>> train.columns
    Index(['id', 'text', 'is_humor', 'votes_no', 'votes_1', 'votes_2', 'votes_3',
           'votes_4', 'votes_5', 'funniness_average'],
          dtype='object')
    >>> train["funniness_average"].mean()
    2.0464498676235694

    ```
    """

    download("haha_2019")

    train_df = pd.read_csv(datapath("haha_2019") / "haha_2019_train.csv")
    test_df = pd.read_csv(datapath("haha_2019") / "haha_2019_test_gold.csv")

    if max_examples is not None:
        train_df = train_df[:max_examples]
        test_df = test_df[:max_examples]

    return train_df, test_df


def load(target="is_humor", max_examples=None):
    """
    Loads the train and test datasets for the [HAHA 2019 corpus](https://www.fing.edu.uy/inco/grupos/pln/haha/index.html#data)
    as lists of texts and target values.

    ##### Arguments

    * `target`: Which column to use for target. Default is `"is_humor"` which can be used for binary classification.
                Another option is `"funniness_average"` which can be used for regression.

    ##### Examples

    Loading with classification targets:

    ```python
    >>> X_train, y_train, X_test, y_test = load()
    >>> print(X_train[13])
    Leí que la falta de sexo trae consigo una notable mejora en el léxico. Me quedo absorto ante tal afirmación carente de raciocinio.
    >>> y_train[13]
    1

    ```

    Loading with regression targets:

    ```python
    >>> X_train, y_train, X_test, y_test = load(target="funniness_average")
    >>> print(X_train[13])
    Leí que la falta de sexo trae consigo una notable mejora en el léxico. Me quedo absorto ante tal afirmación carente de raciocinio.
    >>> y_train[13]
    3.25

    ```

    Loading a subset of the dataset:

    ```python
    >>> Xtrain, Xtest, ytrain, ytest = load(max_examples=100)
    >>> len(Xtrain), len(Xtest), len(ytrain), len(ytest)
    (100, 100, 100, 100)

    ```

    """

    train_df, test_df = load_raw(max_examples)
    X_train = list(train_df["text"])
    y_train = list(train_df[target])
    X_test = list(test_df["text"])
    y_test = list(test_df[target])

    return X_train, np.asarray(y_train), X_test, np.asarray(y_test)
