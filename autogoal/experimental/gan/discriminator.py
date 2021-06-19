from keras.optimizers import adam_v2
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, LeakyReLU, Conv2D
from keras.layers.core import Flatten


class Discriminator:
    def __init__(
            self,
            ERROR='binary_crossentropy',
            LEAKY_SLOPE=0.2,
    ):
        self.model = Sequential()
        self.model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=(128, 128, 3),
                              use_bias=False))
        self.model.add(LeakyReLU(alpha=LEAKY_SLOPE))
        # 64x64x64

        self.model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        self.model.add(BatchNormalization(momentum=0.3))
        self.model.add(LeakyReLU(alpha=LEAKY_SLOPE))
        # 32x32x128

        self.model.add(Conv2D(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        self.model.add(BatchNormalization(momentum=0.3))
        self.model.add(LeakyReLU(alpha=LEAKY_SLOPE))
        # 16x16x256

        self.model.add(Conv2D(512, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        self.model.add(BatchNormalization(momentum=0.3))
        self.model.add(LeakyReLU(alpha=LEAKY_SLOPE))
        # 8x8x512

        self.model.add(Conv2D(1024, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        self.model.add(BatchNormalization(momentum=0.3))
        self.model.add(LeakyReLU(alpha=LEAKY_SLOPE))
        # 4x4x1024

        self.model.add(Flatten())
        self.model.add(Dense(1, activation='sigmoid', use_bias=False))

        self.model.compile(optimizer=adam_v2.Adam(learning_rate=0.0002, beta_1=0.5), loss=ERROR)
