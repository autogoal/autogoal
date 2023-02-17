import numpy as np
import scipy.sparse as sp


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

        a = len(obj) < max(0.1 * original_length, 10)
        b = all(isinstance(x, (str, int, np.int64, np.int32)) for x in obj)
        return a and b
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
