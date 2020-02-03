from autogoal.kb import algorithm, Sentence, List, Word, ContinuousVector, MatrixContinuousDense, Tensor3
from autogoal.grammar import Categorical
from autogoal.utils import nice_repr

import numpy as np


@nice_repr
class VectorAggregator:
    def __init__(self, mode: Categorical("mean", "max")):
        self.mode = mode

    def run(self, input: List(ContinuousVector())) -> ContinuousVector():
        input = np.vstack(input)

        if self.mode == "mean":
            return input.mean(axis=1)
        elif self.mode == "max":
            return input.max(axis=1)

        raise ValueError("Invalid mode: %s" % self.mode)


@nice_repr
class MatrixBuilder:
    """
    Builds a matrix from a list of vectors.

    ##### Examples

    ```python
    >>> import numpy as np
    >>> x1 = np.asarray([1,2,3])
    >>> x2 = np.asarray([2,3,4])
    >>> x3 = np.asarray([3,4,5])
    >>> MatrixBuilder().run([x1, x2, x3])
    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])

    ```
    """
    def run(self, input: List(ContinuousVector())) -> MatrixContinuousDense():
        return np.vstack(input)


@nice_repr
class TensorBuilder:
    """
    Builds a 3D tensor from a list of matrices.

    ##### Examples

    ```python
    >>> import numpy as np
    >>> x1 = np.asarray([[1,2],[3,4]])
    >>> x2 = np.asarray([[2,3],[4,5]])
    >>> x3 = np.asarray([[3,4],[5,6]])
    >>> TensorBuilder().run([x1, x2, x3])
    array([[[1, 2],
            [3, 4]],
    <BLANKLINE>
           [[2, 3],
            [4, 5]],
    <BLANKLINE>
           [[3, 4],
            [5, 6]]])

    ```
    """
    def run(self, input: List(MatrixContinuousDense())) -> Tensor3():
        return np.vstack([np.expand_dims(m, axis=0) for m in input])
