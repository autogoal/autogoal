from tensorflow.keras.layers import Conv2DTranspose as _Conv2DTranspose
from tensorflow.keras.layers import Conv2D as _Conv2D
from tensorflow.keras import regularizers

from autogoal.utils import nice_repr
from autogoal.grammar._cfg import DiscreteValue, ContinuousValue, CategoricalValue



@nice_repr
class Conv2DTranspose(_Conv2DTranspose):
    def __init__(
            self,
            filters: DiscreteValue(2, 8),
            kernel_size: CategoricalValue(3, 5, 7),
            l1: ContinuousValue(0, 1e-3),
            l2: ContinuousValue(0, 1e-3),
    ):
        self.l1 = l1
        self.l2 = l2
        super().__init__(
            filters=2 ** filters,
            kernel_size=(kernel_size, kernel_size),
            kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
            padding="same",
            data_format="channels_last",
        )

@nice_repr
class Conv2D(_Conv2D):
    def __init__(
        self,
        filters: DiscreteValue(2, 8),
        kernel_size: CategoricalValue(3, 5, 7),
        l1: ContinuousValue(0, 1e-3),
        l2: ContinuousValue(0, 1e-3),
        activation:CategoricalValue(None, 'relu'),
    ):
        self.l1 = l1
        self.l2 = l2
        super().__init__(
            filters=2 ** filters,
            kernel_size=(kernel_size, kernel_size),
            kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
            padding="same",
            data_format="channels_last",
            activation=activation
        )

