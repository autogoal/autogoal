#

import warnings

import numpy as np
import scipy.sparse as sp
from typing import List, Tuple

from autogoal.contrib._utils import (
    is_categorical,
    is_continuous,
    is_matrix_continuous_dense,
    is_matrix_continuous_sparse,
    is_string_list,
    is_discrete
)


class String(str):
    def __new__(cls, x):
        return str.__new__(cls, x)


def _get_class_name(cls):
    return str(cls).split(".")[-1]


def combine_types(*types):
    if len(types) == 1:
        return types[0]

    types = set(types)

    if types == {kb.MatrixContinuousDense, kb.MatrixContinuousSparse}:
        return kb.MatrixContinuous

    return None


from autogoal import kb

DATA_RESOLVERS = {
    kb.MatrixContinuousDense: is_matrix_continuous_dense,
    kb.MatrixContinuousSparse: is_matrix_continuous_sparse,
    kb.VectorCategorical: is_categorical,
    kb.VectorContinuous: is_continuous,
    kb.Seq[kb.Sentence]: is_string_list,
}


DATA_TYPE_EXAMPLES = {
    kb.MatrixContinuousDense: np.random.rand(10, 10),
    kb.MatrixContinuousSparse: sp.rand(10, 10),
    kb.VectorCategorical: np.asarray(["A"] * 5 + ["B"] * 5),
    kb.VectorContinuous: np.random.rand(10),
    kb.VectorDiscrete: np.random.randint(0, 10, (10,), dtype=int),
    kb.Seq[kb.Sentence]: ["abc bcd def feg geh hij jkl lmn nop pqr"] * 10,
}


def is_algorithm(cls, verbose=False):
    if hasattr(cls, "fit") and hasattr(cls, "predict"):
        return "estimator"
    else:
        if verbose:
            warnings.warn("%r doesn't have `fit`" % cls)

    if hasattr(cls, "transform"):
        return "transformer"
    else:
        if verbose:
            warnings.warn("%r doesn't have `transform`" % cls)

    return False


def is_classifier(cls, verbose=False):
    """Determine if `cls` corresponds to something that resembles an sklearn classifier.
    If True, returns the valid (input, output) types.

    Examples:

    >>> from sklearn.linear_model import LogisticRegression, LinearRegression
    >>> is_classifier(LogisticRegression)
    (True, (Tuple(MatrixContinuous(), CategoricalVector()), CategoricalVector()))
    >>> is_classifier(LinearRegression)
    (False, None)

    """
    if not is_algorithm(cls, verbose=verbose):
        return False, None

    inputs = []

    for input_type in [kb.MatrixContinuousDense, kb.MatrixContinuousSparse]:
        try:
            X = DATA_TYPE_EXAMPLES[input_type]
            y = DATA_TYPE_EXAMPLES[kb.VectorCategorical]

            clf = cls()
            clf.fit(X, y)
            y = clf.predict(X)

            assert is_categorical(y)
            inputs.append(input_type)
        except Exception as e:
            if verbose:
                warnings.warn(str(e))

    inputs = combine_types(*inputs)

    if inputs:
        return True, ((inputs, kb.VectorCategorical), kb.VectorCategorical)
    else:
        return False, None


def is_regressor(cls, verbose=False):
    """Determine if `cls` corresponds to something that resembles an sklearn regressor.
    If True, returns the valid (input, output) types.

    Examples:

    >>> from sklearn.linear_model import LogisticRegression, LinearRegression
    >>> is_regressor(LogisticRegression)
    (False, None)
    >>> is_regressor(LinearRegression)
    (True, (Tuple(MatrixContinuous(), ContinuousVector()), ContinuousVector()))

    """
    if not is_algorithm(cls, verbose=verbose):
        return False, None

    inputs = []

    for input_type in [kb.MatrixContinuousDense, kb.MatrixContinuousSparse]:
        try:
            X = DATA_TYPE_EXAMPLES[input_type]
            y = DATA_TYPE_EXAMPLES[kb.VectorContinuous]

            clf = cls()
            clf.fit(X, y)
            y = clf.predict(X)

            assert is_continuous(y)
            inputs.append(input_type)
        except Exception as e:
            if verbose:
                warnings.warn(str(e))

    inputs = combine_types(*inputs)

    if inputs:
        return True, ((inputs, kb.VectorContinuous), kb.VectorContinuous)
    else:
        return False, None


def is_clusterer(cls, verbose=False):
    """Determine if `cls` corresponds to something that resembles an sklearn clustering algorithm.
    If True, returns the valid (input, output) types.

    Examples:

    >>> from sklearn.linear_model import LogisticRegression, LinearRegression
    >>> is_clusterer(LogisticRegression)
    (False, None)
    >>> is_clusterer(LinearRegression)
    (False, None)
    >>> from sklearn.cluster import KMeans
    >>> is_clusterer(KMeans)
    (True, (MatrixContinuous(), DiscreteVector()))

    """
    if not is_algorithm(cls, verbose=verbose):
        return False, None

    inputs = []

    for input_type in [kb.MatrixContinuousDense, kb.MatrixContinuousSparse]:
        try:
            X = DATA_TYPE_EXAMPLES[input_type]

            clf = cls()
            y = clf.fit_predict(X)

            assert is_discrete(y)
            inputs.append(input_type)
        except Exception as e:
            if verbose:
                warnings.warn(str(e))

    inputs = combine_types(*inputs)

    if inputs:
        return True, (inputs, kb.VectorDiscrete)
    else:
        return False, None


def is_transformer(cls, verbose=False):
    """Determine if `cls` corresponds to something that resembles an sklearn general transformer.
    If True, returns the valid (input, output) types.

    Examples:

    >>> from sklearn.feature_extraction.text import CountVectorizer
    >>> is_transformer(CountVectorizer)
    (True, (List(Sentence()), MatrixContinuousSparse()))
    >>> from sklearn.decomposition.pca import PCA
    >>> is_transformer(PCA)
    (True, (MatrixContinuousDense(), MatrixContinuousDense()))

    """
    if not is_algorithm(cls, verbose=verbose):
        return False, None

    allowed_inputs = set()
    allowed_outputs = set()

    for input_type in [
        kb.MatrixContinuousDense,
        kb.MatrixContinuousSparse,
        kb.Seq[kb.Sentence],
    ]:
        for output_type in [
            kb.MatrixContinuousDense,
            kb.MatrixContinuousSparse,
            kb.Seq[kb.Sentence],
        ]:
            try:
                X = DATA_TYPE_EXAMPLES[input_type]

                clf = cls()
                X = clf.fit_transform(X)

                assert is_data_type(X, output_type)

                allowed_inputs.add(input_type)
                allowed_outputs.add(output_type)
            except Exception as e:
                if verbose:
                    warnings.warn(str(e))

    if len(allowed_outputs) != 1:
        return False, None

    inputs = combine_types(*allowed_inputs)

    if allowed_inputs:
        return True, (inputs, list(allowed_outputs)[0])
    else:
        return False, None


def is_data_type(X, data_type):
    return DATA_RESOLVERS[data_type](X)


IO_TYPE_HANDLER = [is_classifier, is_regressor, is_clusterer, is_transformer]


def get_input_output(cls, verbose=False):
    for func in IO_TYPE_HANDLER:
        matches, types = func(cls, verbose=verbose)
        if matches:
            return types

    return None, None


def solve_type(obj):
    for type_, resolver in DATA_RESOLVERS.items():
        if resolver(obj):
            return type_

    raise ValueError("Unresolved type for %r" % obj)
