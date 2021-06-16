from tensorflow.keras.layers import Conv2D as _Conv2D
from tensorflow.keras.layers import Conv2DTranspose as _Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D as _MaxPooling2D
from autogoal.utils import nice_repr
from autogoal.grammar import DiscreteValue


def conv_2d_x(filters: int):
    @nice_repr
    class Conv2D(_Conv2D):
        def __init__(self):
            super().__init__(filters=filters, kernel_size=3, activation="relu")

    return Conv2D


def conv_transpose_x(filters: int):
    @nice_repr
    class Conv2DTranspose(_Conv2DTranspose):
        def __init__(self):
            super().__init__(filters=filters // 2, kernel_size=2, stride=2)

    return Conv2DTranspose


@nice_repr
class OutConv2D(_Conv2D):
    def __init__(self):
        super().__init__(filters=1, kernel_size=1)


@nice_repr
class MaxPooling2D(_MaxPooling2D):
    def __init__(self):
        super().__init__(pool_size=2, strides=2)
