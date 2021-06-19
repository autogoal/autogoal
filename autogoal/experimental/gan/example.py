from autogoal.ml import AutoML
from autogoal.kb import *
from autogoal.search import RichLogger
from autogoal.utils import Min, Gb
from autogoal.experimental.gan.utils import ImageFile
from autogoal.experimental.gan.gan import (
    GAN,
)
from autogoal.contrib import find_classes
from autogoal.experimental.gan.dataset import load

import matplotlib.pyplot as plt


def main():
    automl = AutoML(
        input=(Seq[ImageFile],),
        output=ImageFile,
        registry=[GAN] + find_classes(),
        evaluation_timeout=Min,
        memory_limit=4 * Gb,
        search_timeout=Min,
    )

    x_train = load()

    automl.fit(x_train, x_train, logger=[RichLogger()])

    image = automl.generate(x_train)

    plt.figure(figsize=(10, 10))
    plt.imshow(image.astype('uint8'), interpolation='nearest')
    plt.axis('off')
    plt.waitforbuttonpress()


if __name__ == '__main__':
    main()
