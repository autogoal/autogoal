from autogoal.kb import Matrix, MatrixContinuous, MatrixContinuousDense, MatrixContinuousSparse
from autogoal.kb import conforms
from autogoal.kb import algorithm


def test_matrix_hierarchy():
    assert conforms(MatrixContinuous(), Matrix())
    assert conforms(MatrixContinuousSparse(), Matrix())
    assert conforms(MatrixContinuousDense(), Matrix())
    assert conforms(MatrixContinuousSparse(), MatrixContinuous())
    assert conforms(MatrixContinuousDense(), MatrixContinuous())


class ExactAlgorithm:
    def run(self, input:MatrixContinuousDense()) -> MatrixContinuousDense():
        pass


class HigherInputAlgorithm:
    def run(self, input:MatrixContinuous()) -> MatrixContinuousDense():
        pass


class LowerOutputAlgorithm:
    def run(self, input:MatrixContinuousDense()) -> MatrixContinuousDense():
        pass


def test_polimorphic_interface():
    interface = algorithm(MatrixContinuousDense(), MatrixContinuousDense())
    assert interface.is_compatible(ExactAlgorithm)
    assert interface.is_compatible(HigherInputAlgorithm)

    interface = algorithm(MatrixContinuousDense(), MatrixContinuous())
    assert interface.is_compatible(LowerOutputAlgorithm)
