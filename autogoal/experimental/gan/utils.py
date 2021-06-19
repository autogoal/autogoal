from datetime import datetime
import os
from os import path

from imageio import imwrite

from autogoal.kb import SemanticType
import numpy as np


class ImageFile(SemanticType):
    """Semantic type for image files"""

    @classmethod
    def _match(cls, x):
        # common image types
        image_types = ['tif', 'tiff', 'bmp', 'jpg', 'jpeg', 'gif', 'png', 'eps', ]
        try:
            return path.isfile(x) and x.split(".")[-1] in image_types
        except TypeError:
            return False


class ConsoleColors:
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'


def image_generator(generator, n_images):
    example_path = './example'
    noise = np.random.normal(0, 1, [n_images, 100])
    generated_images = generator.predict(noise)
    generated_images.reshape(n_images, 128, 128, 3)
    generated_images = generated_images * 127.5 + 127.5
    generated_images.astype('uint8')
    for i in range(n_images):
        imwrite(os.path.join(example_path, 'example_' + str(datetime.now()) + '.png'),
                generated_images[i].reshape(128, 128, 3))
