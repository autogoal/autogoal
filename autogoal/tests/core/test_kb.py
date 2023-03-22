from autogoal.kb import (
    Matrix,
    MatrixContinuous,
    MatrixContinuousDense,
    MatrixContinuousSparse,
)
from autogoal.kb import AlgorithmBase


def test_matrix_hierarchy():
    assert issubclass(MatrixContinuous, Matrix)
    assert issubclass(MatrixContinuousSparse, Matrix)
    assert issubclass(MatrixContinuousDense, Matrix)
    assert issubclass(MatrixContinuousSparse, MatrixContinuous)
    assert issubclass(MatrixContinuousDense, MatrixContinuous)


class ExactAlgorithm(AlgorithmBase):
    def run(self, input: MatrixContinuousDense) -> MatrixContinuousDense:
        pass


class HigherInputAlgorithm(AlgorithmBase):
    def run(self, input: MatrixContinuous) -> MatrixContinuousDense:
        pass


class LowerOutputAlgorithm(AlgorithmBase):
    def run(self, input: MatrixContinuousDense) -> MatrixContinuousDense:
        pass


def test_exact_compatibilty():
    assert ExactAlgorithm.is_compatible_with([MatrixContinuousDense])


def test_subtype_compatibility():
    assert HigherInputAlgorithm.is_compatible_with([MatrixContinuousDense])
