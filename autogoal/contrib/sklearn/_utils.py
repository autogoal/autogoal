#

import warnings

import numpy as np
import scipy.sparse as sp
from owlready2 import Not


class String(str):
    def __new__(cls, x):
        return str.__new__(cls, x)


def _get_class_name(cls):
    return str(cls).split(".")[-1]


def get_data_for(*classes, name=None):
    sorted_classes = tuple(sorted(classes, key=_get_class_name))
    instance_name = name or "".join(_get_class_name(cls) for cls in sorted_classes)

    if len(classes) == 1:
        instance_name += "Instance"

    solved = onto[instance_name]

    if solved:
        return solved

    instance = classes[0](instance_name)
    instance.is_a.extend(classes[1:])

    return instance


def combine_types(*types):
    if len(types) == 1:
        return types[0]

    types = set(types)

    if types == {MatrixContinuousDense, MatrixContinuousSparse}:
        return MatrixContinuous

    return None


def is_matrix_continuous_dense(obj):
    """Determine if `obj` is a continuous dense matrix (i.e., NxM floats).

    Examples:

    >>> is_matrix_continuous_dense([[0, 1],[0, 1]])
    True
    >>> is_matrix_continuous_dense([[0,1]])
    True
    >>> is_matrix_continuous_dense([0,1])
    False
    >>> is_matrix_continuous_dense(np.random.rand(10,10))
    True
    >>> is_matrix_continuous_dense(np.random.rand(10))
    False

    """
    try:
        obj = np.asarray(obj)
        return len(obj.shape) == 2
    except:
        return False


def is_matrix_continuous_sparse(obj):
    """Determine if `obj` is a continuous sparse matrix (i.e., NxM floats).

    Examples:

    >>> is_matrix_continuous_sparse(sp.rand(10,10))
    True
    >>> is_matrix_continuous_sparse([[0,1]])
    False
    >>> is_matrix_continuous_sparse([0,1])
    False
    >>> is_matrix_continuous_sparse(np.random.rand(10,10))
    False
    >>> is_matrix_continuous_sparse(np.random.rand(10))
    False

    """
    try:
        sp.rand
        return sp.issparse(obj) and len(obj.shape) == 2
    except:
        return False


def is_categorical(obj):
    """Determines if `obj` is a sequence of categories (integer or string)

    Examples:

    >>> is_categorical(['A'] * 5 + ['B'] * 5)
    True
    >>> is_categorical(np.asarray(list('ABCABCABC')))
    True

    """
    try:
        obj = np.asarray(obj)
        assert len(obj.shape) == 1

        original_length = len(obj)
        obj = set(obj)

        return len(obj) < max(0.1 * original_length, 10) and all(
            isinstance(x, (str, int)) for x in obj
        )
    except:
        return False


def is_continuous(obj):
    """Determines if `obj` is a sequence of float values

    Examples:

    >>> is_continuous(np.random.rand(10))
    True
    >>> is_continuous(np.random.rand(10,10))
    False

    """
    try:
        obj = np.asarray(obj)
        assert len(obj.shape) == 1

        return not all(obj.astype(int) == obj)
    except:
        return False


def is_discrete(obj):
    """Determines if `obj` is a sequence of integer values

    Examples:

    >>> is_discrete(np.random.randint(0,1,(10,)))
    True
    >>> is_discrete(np.random.rand(10))
    False

    """
    try:
        obj = np.asarray(obj)
        assert len(obj.shape) == 1

        return all(obj.astype(int) == obj)
    except:
        return False


def is_string_list(obj):
    """Determines if `obj` is a sequence of strings.

    Examples:

    >>> is_string_list(['hello world', 'another sentence'])
    True
    >>> is_string_list(np.random.rand(10))
    False

    """
    try:
        obj = np.asarray(obj)
        assert len(obj.shape) == 1

        original_length = len(obj)
        obj = set(obj)

        return len(obj) > 0.1 * original_length and all(isinstance(x, str) for x in obj)
    except:
        return False


DATA_RESOLVERS = {
    MatrixContinuousDense: is_matrix_continuous_dense,
    MatrixContinuousSparse: is_matrix_continuous_sparse,
    CategoricalVector: is_categorical,
    ContinuousVector: is_continuous,
    StringList: is_string_list,
}


DATA_TYPE_EXAMPLES = {
    MatrixContinuousDense: np.random.rand(10, 10),
    MatrixContinuousSparse: sp.rand(10, 10),
    CategoricalVector: np.asarray(["A"] * 5 + ["B"] * 5),
    ContinuousVector: np.random.rand(10),
    DiscreteVector: np.random.randint(0, 10, (10,), dtype=int),
    StringList: ["abc bcd def feg geh hij jkl lmn nop pqr"] * 10,
}


def is_algorithm(cls, verbose=False):
    if hasattr(cls, "fit"):
        return True
    else:
        if verbose:
            warnings.warn("%r doesn't have `fit`" % cls)

    if hasattr(cls, "transform"):
        return True
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
    (ontoml.ContinuousMatrix, ontoml.CategoricalVector)
    >>> is_classifier(LinearRegression)
    False

    """
    if not is_algorithm(cls, verbose=verbose):
        return False

    inputs = []

    for input_type in [MatrixContinuousDense, MatrixContinuousSparse]:
        try:
            X = DATA_TYPE_EXAMPLES[input_type]
            y = DATA_TYPE_EXAMPLES[CategoricalVector]

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
        return (inputs, CategoricalVector)
    else:
        return False


def is_regressor(cls, verbose=False):
    """Determine if `cls` corresponds to something that resembles an sklearn regressor.
    If True, returns the valid (input, output) types.

    Examples:

    >>> from sklearn.linear_model import LogisticRegression, LinearRegression
    >>> is_regressor(LogisticRegression)
    False
    >>> is_regressor(LinearRegression)
    (ontoml.ContinuousMatrix, ontoml.ContinuousVector)

    """
    if not is_algorithm(cls, verbose=verbose):
        return False

    inputs = []

    for input_type in [MatrixContinuousDense, MatrixContinuousSparse]:
        try:
            X = DATA_TYPE_EXAMPLES[input_type]
            y = DATA_TYPE_EXAMPLES[ContinuousVector]

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
        return (inputs, ContinuousVector)
    else:
        return False


def is_clusterer(cls, verbose=False):
    """Determine if `cls` corresponds to something that resembles an sklearn clustering algorithm.
    If True, returns the valid (input, output) types.

    Examples:

    >>> from sklearn.linear_model import LogisticRegression, LinearRegression
    >>> is_clusterer(LogisticRegression)
    False
    >>> is_clusterer(LinearRegression)
    False
    >>> from sklearn.cluster import KMeans
    >>> is_clusterer(KMeans)
    (ontoml.ContinuousMatrix, ontoml.DiscreteVector)

    """
    if not is_algorithm(cls, verbose=verbose):
        return False

    inputs = []

    for input_type in [MatrixContinuousDense, MatrixContinuousSparse]:
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
        return (inputs, DiscreteVector)
    else:
        return False


def is_transformer(cls, verbose=False):
    """Determine if `cls` corresponds to something that resembles an sklearn general transformer.
    If True, returns the valid (input, output) types.

    Examples:

    >>> from sklearn.feature_extraction.text import CountVectorizer
    >>> is_transformer(CountVectorizer)
    (ontoml.StringList, ontoml.ContinuousMatrixSparse)
    >>> from sklearn.decomposition.pca import PCA
    >>> is_transformer(PCA)
    (ontoml.ContinuousDenseMatrix, ontoml.ContinuousDenseMatrix)

    """
    if not is_algorithm(cls, verbose=verbose):
        return False

    allowed_inputs = set()
    allowed_outputs = set()

    for input_type in [MatrixContinuousDense, MatrixContinuousSparse, StringList]:
        for output_type in [MatrixContinuousDense, MatrixContinuousSparse, StringList]:
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
        return False

    inputs = combine_types(*allowed_inputs)

    if allowed_inputs:
        return (inputs, list(allowed_outputs)[0])
    else:
        return False


def is_data_type(X, data_type):
    return DATA_RESOLVERS[data_type](X)

IO_TYPE_HANDLER = [
    is_classifier,
    is_regressor,
    is_clusterer,
    is_transformer
]

def get_input_output(cls, verbose=False):
    for func in IO_TYPE_HANDLER:
        matches, types = func(cls, verbose=verbose)
        if matches:
            return types
    return None

def solve_type(obj):
    for type_, resolver in DATA_RESOLVERS.items():
        if resolver(obj):
            return type_

    raise ValueError("Unresolved type for %r" % obj)
