from keras.optimizers import adam_v2
from keras.models import Sequential
from keras.layers import Dense, Reshape, Conv2DTranspose, BatchNormalization, LeakyReLU
from keras.layers.core import Activation


class Generator:
    def __init__(
            self,
            TAM_IN=100,
            ERROR='binary_crossentropy',
            LEAKY_SLOPE=0.2,
    ):
        self.model = Sequential()
        self.model.add(Dense(1024 * 4 * 4, use_bias=False, input_shape=(TAM_IN,)))
        self.model.add(BatchNormalization(momentum=0.3))
        self.model.add(LeakyReLU(alpha=LEAKY_SLOPE))
        self.model.add(Reshape((4, 4, 1024)))
        # 4x4x1024

        self.model.add(Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        self.model.add(BatchNormalization(momentum=0.3))
        self.model.add(LeakyReLU(alpha=LEAKY_SLOPE))
        # 8x8x512

        self.model.add(Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        self.model.add(BatchNormalization(momentum=0.3))
        self.model.add(LeakyReLU(alpha=LEAKY_SLOPE))
        # 16x16x256

        self.model.add(Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        self.model.add(BatchNormalization(momentum=0.3))
        self.model.add(LeakyReLU(alpha=LEAKY_SLOPE))
        # 32x32x128

        self.model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        self.model.add(BatchNormalization(momentum=0.3))
        self.model.add(LeakyReLU(alpha=LEAKY_SLOPE))
        # 64x64x64

        self.model.add(Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        self.model.add(Activation('tanh'))
        # 128x128x3

        self.model.compile(optimizer=adam_v2.Adam(learning_rate=0.0002, beta_1=0.5), loss=ERROR)
