from tensorflow.keras.layers import Conv1D as _Conv1D
from tensorflow.keras.layers import GlobalMaxPooling1D as _GlobalMaxPooling1D
from tensorflow.keras.layers import GlobalAveragePooling1D as _GlobalAveragePooling1D
from tensorflow.keras.layers import Conv2D as _Conv2D
from tensorflow.keras.layers import SeparableConv2D as _SeparableConv2D
from tensorflow.keras.layers import MaxPool2D as _MaxPool2D
from tensorflow.keras.layers import GlobalAveragePooling2D as _GlobalAveragePooling2D
from tensorflow.keras.layers import Dense as _Dense
from tensorflow.keras.layers import Embedding as _Embedding
from tensorflow.keras.layers import LSTM as _LSTM
from tensorflow.keras.layers import BatchNormalization as _BatchNormalization
from tensorflow.keras.layers import Activation as _Activation
from tensorflow.keras.layers import Dropout as _Dropout
from tensorflow.keras.layers import TimeDistributed as _TimeDistributed
from tensorflow.keras.layers import Flatten as _Flatten
from tensorflow.keras.layers import Reshape, Bidirectional

from autogoal.grammar import Boolean, Categorical, Discrete, Continuous
from autogoal.utils import nice_repr


@nice_repr
class Seq2SeqLSTM(_LSTM):
    def __init__(
        self,
        units: Discrete(32, 1024),
        activation: Categorical("tanh", "sigmoid", "relu", "linear"),
        recurrent_activation: Categorical("tanh", "sigmoid", "relu", "linear"),
        dropout: Continuous(0, 0.5),
        recurrent_dropout: Continuous(0, 0.5),
    ):
        super().__init__(
            units=units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=True,
        )


@nice_repr
class Seq2VecLSTM(_LSTM):
    def __init__(
        self,
        units: Discrete(32, 1024),
        activation: Categorical("tanh", "sigmoid", "relu", "linear"),
        recurrent_activation: Categorical("tanh", "sigmoid", "relu", "linear"),
        dropout: Continuous(0, 0.5),
        recurrent_dropout: Continuous(0, 0.5),
    ):
        super().__init__(
            units=units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=False,
        )


@nice_repr
class Seq2SeqBiLSTM(Bidirectional):
    def __init__(
        self,
        merge_mode: Categorical("sum", "mul", "concat", "ave"),
        units: Discrete(32, 1024),
        activation: Categorical("tanh", "sigmoid", "relu", "linear"),
        recurrent_activation: Categorical("tanh", "sigmoid", "relu", "linear"),
        dropout: Continuous(0, 0.5),
        recurrent_dropout: Continuous(0, 0.5),
    ):
        super().__init__(
            layer=_LSTM(
                units=units,
                activation=activation,
                recurrent_activation=recurrent_activation,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                return_sequences=True,
            ),
            merge_mode=merge_mode,
        )


@nice_repr
class Seq2VecBiLSTM(Bidirectional):
    def __init__(
        self,
        merge_mode: Categorical("sum", "mul", "concat", "ave"),
        units: Discrete(32, 1024),
        activation: Categorical("tanh", "sigmoid", "relu", "linear"),
        recurrent_activation: Categorical("tanh", "sigmoid", "relu", "linear"),
        dropout: Continuous(0, 0.5),
        recurrent_dropout: Continuous(0, 0.5),
    ):
        super().__init__(
            layer=_LSTM(
                units=units,
                activation=activation,
                recurrent_activation=recurrent_activation,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                return_sequences=False,
            ),
            merge_mode=merge_mode,
        )


# @nice_repr
# class Reshape2D(Reshape):
#     def __init__(self):
#         super().__init__(target_shape=(-1, 1))


# @nice_repr
# class Embedding(_Embedding):
#     def __init__(self, output_dim: Discrete(32, 128)):
#         super().__init__(input_dim=1000, output_dim=output_dim)


@nice_repr
class Dense(_Dense):
    def __init__(self, units: Discrete(128, 1024), **kwargs):
        super().__init__(units=units, **kwargs)


@nice_repr
class Conv1D(_Conv1D):
    def __init__(
        self,
        filters: Discrete(2, 9),
        strides: Categorical(1, 2),
        kernel_size: Discrete(1, 30),
        padding: Categorical("valid", "causal"),
    ):
        super().__init__(
            filters=2 ** filters,
            strides=strides,
            kernel_size=kernel_size,
            padding=padding,
        )


@nice_repr
class GlobalMaxPooling1D(_GlobalMaxPooling1D):
    def __init__(self):
        super().__init__(data_format="channels_last")


@nice_repr
class GlobalAveragePooling1D(_GlobalAveragePooling1D):
    def __init__(self):
        super().__init__(data_format="channels_last")


@nice_repr
class Conv2D(_Conv2D):
    def __init__(
        self,
        filters: Discrete(2, 9),
        strides: Categorical(1, 2),
        kernel_size: Categorical(3, 5, 7),
    ):
        super().__init__(
            filters=2 ** filters, strides=strides, kernel_size=kernel_size, padding="same"
        )


@nice_repr
class SeparableConv2D(_SeparableConv2D):
    def __init__(
        self,
        filters: Discrete(2, 9),
        strides: Categorical(1, 2),
        kernel_size: Categorical(3, 5, 7),
    ):
        super().__init__(
            filters=2 ** filters, strides=strides, kernel_size=kernel_size, padding="same"
        )


@nice_repr
class MaxPooling2D(_MaxPool2D):
    def __init__(self, pool_size: Categorical(2, 3)):
        super().__init__(pool_size=pool_size, padding="same")


@nice_repr
class GlobalAveragePooling2D(_GlobalAveragePooling2D):
    def __init__(self):
        super().__init__(data_format="channels_last")


@nice_repr
class BatchNormalization(_BatchNormalization):
    def __init__(self):
        super().__init__()


@nice_repr
class Dropout(_Dropout):
    def __init__(self, rate: Continuous(0, 0.5)):
        super().__init__(rate=rate)


@nice_repr
class Activation(_Activation):
    def __init__(
        self,
        function: Categorical(
            "elu",
            "selu",
            "relu",
            "tanh",
            "sigmoid",
            "hard_sigmoid",
            "exponential",
            "linear",
        ),
    ):
        self.function = function
        super().__init__(activation=function)


@nice_repr
class Flatten(_Flatten):
    def __init__(self):
        super().__init__()


def TimeDistributed(Layer):
    @nice_repr
    class TimeDistributed(_TimeDistributed):
        def __init__(self, layer: Layer):
            super().__init__(layer)

    return TimeDistributed
