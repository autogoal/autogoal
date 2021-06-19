import os

from imageio import imwrite
from keras import Sequential
from keras.optimizers import adam_v2

from autogoal.experimental.gan.discriminator import Discriminator
from autogoal.experimental.gan.generator import Generator
from autogoal.experimental.gan.utils import ImageFile
from autogoal.kb import AlgorithmBase
from autogoal.grammar import DiscreteValue
from autogoal.kb._semantics import Seq

import numpy as np
np.random.seed(5)


class GAN(AlgorithmBase):
    def __init__(
            self,
            n_components: DiscreteValue(min=1, max=20),
            n_iter: DiscreteValue(min=3000, max=4000),
            covariance_type="diag",
            ERROR='binary_crossentropy',
            TAM_LOTE=128,
            TAM_IN=100
    ):
        self.TAM_IN = TAM_IN
        self.TAM_LOTE = TAM_LOTE
        self._mode = "train"
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.models = {}
        self.model = Sequential()
        self.model.add(self.generator.model)
        self.discriminator.model.trainable = False
        self.model.add(self.discriminator.model)
        self.model.compile(optimizer=adam_v2.Adam(learning_rate=0.0002, beta_1=0.5), loss=ERROR)

        super().__init__()

    def train(self):
        self._mode = "train"

    def eval(self):
        self._mode = "eval"

    def run(self, x: Seq[ImageFile], y: Seq[ImageFile]) -> ImageFile:
        if self._mode == "train":
            self._train(x)
            return None
        else:
            return self._eval(x)

    def _train(self, x_train: Seq[ImageFile]):
        # Training
        for i in range(1, self.n_iter + 1):
            print("Epoch " + str(i))

            # Create batch of real images and another with fake images
            noise = np.random.normal(0, 1, [self.TAM_LOTE, self.TAM_IN])
            fake_batch = self.generator.model.predict(noise)

            idx = np.random.randint(low=0, high=x_train.shape[0], size=self.TAM_LOTE)
            real_batch = x_train[idx]

            # Train discriminator with fake and real images, in every case calculate error
            self.discriminator.model.trainable = True

            d_real_errors = self.discriminator.model.train_on_batch(real_batch,
                                                                    np.ones(self.TAM_LOTE) * 0.9)
            d_fake_errors = self.discriminator.model.train_on_batch(fake_batch,
                                                                    np.zeros(self.TAM_LOTE) * 0.1)

            self.discriminator.model.trainable = False

            # Train GAN: Will be generated aleatory data and will be presented to GAN as real
            noise = np.random.normal(0, 1, [self.TAM_LOTE, self.TAM_IN])
            gan_error = self.model.train_on_batch(noise, np.ones(self.TAM_LOTE))

            # Save Generator
            if i == 1 or i % 1000 == 0:
                self.generator.model.save('generador.h5')

        return None

    def _eval(self, x: Seq[ImageFile]) -> ImageFile:
        example_path = './'
        noise = np.random.normal(0, 1, [1, 100])
        generated_images = self.generator.model.predict(noise)
        generated_images.reshape(1, 128, 128, 3)
        generated_images = generated_images * 127.5 + 127.5
        generated_images.astype('uint8')
        imwrite(os.path.join(example_path, 'example_0.png'), generated_images[0].reshape(128, 128, 3))
        return generated_images[0].reshape(128, 128, 3)
