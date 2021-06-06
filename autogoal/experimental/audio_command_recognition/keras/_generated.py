from tensorflow.keras.layers import Conv1D as _Conv1D
from tensorflow.keras.layers import MaxPooling1D as _MaxPooling1D
from autogoal.grammar import DiscreteValue, CategoricalValue
from autogoal.utils import nice_repr


@nice_repr
class Conv1D(_Conv1D):
    def __init__(
        self,
        filters: DiscreteValue(2, 8),
        kernel_size: CategoricalValue(3, 5, 7),
        activation: CategoricalValue(None, "relu"),
    ):
        super().__init__(
            filters=2 ** filters,
            kernel_size=kernel_size,
            padding="causal",
            activation=activation,
        )


@nice_repr
class MaxPooling1D(_MaxPooling1D):
    def __init__(self, pool_size: DiscreteValue(2, 8), strides: DiscreteValue(1, 2)):
        super().__init__(pool_size=pool_size, strides=strides, padding="same")

