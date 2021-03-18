from autogoal.kb import Matrix, MatrixContinuous, MatrixContinuousDense, MatrixContinuousSparse
from autogoal.kb import algorithm


def test_matrix_hierarchy():
    assert issubclass(MatrixContinuous, Matrix)
    assert issubclass(MatrixContinuousSparse, Matrix)
    assert issubclass(MatrixContinuousDense, Matrix)
    assert issubclass(MatrixContinuousSparse, MatrixContinuous)
    assert issubclass(MatrixContinuousDense, MatrixContinuous)


class ExactAlgorithm:
    def run(self, input:MatrixContinuousDense) -> MatrixContinuousDense:
        pass


class HigherInputAlgorithm:
    def run(self, input:MatrixContinuous) -> MatrixContinuousDense:
        pass


class LowerOutputAlgorithm:
    def run(self, input:MatrixContinuousDense) -> MatrixContinuousDense:
        pass


# NOTE: This functionality doesn't currently exist
# def test_polimorphic_interface():
#     interface = algorithm(MatrixContinuousDense, MatrixContinuousDense)
#     assert interface.is_compatible(ExactAlgorithm)
#     assert interface.is_compatible(HigherInputAlgorithm)

#     interface = algorithm(MatrixContinuousDense, MatrixContinuous)
#     assert interface.is_compatible(LowerOutputAlgorithm)
