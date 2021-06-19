from imageio import imread
from autogoal.datasets import download_and_save, datapath, unpack
import numpy as np
import os

from autogoal.experimental.progressive_gan.utils import ConsoleColors

data_url = "https://github.com/AntonioJesus0398/speech-recognition-dataset/raw/master/speech-recognition-dataset.zip"

dataset = "image-dataset"


def load():
    # Download the data
    file_name = f"{dataset}.zip"
    file_path = datapath(file_name)
    if not file_path.exists():
        print(f"{ConsoleColors.WARNING}Downloading {dataset}...")
        download_and_save(data_url, file_path, True)
        print(f"{ConsoleColors.OKGREEN}Done!\n")

    # Unpack the zip file
    dir_path = datapath(dataset)
    if not dir_path.exists():
        print(f"{ConsoleColors.WARNING}Unpacking {file_name}...")
        unpack(file_name)
        print(f"{ConsoleColors.OKGREEN}Done.\n")

    # Build training set
    file_list = os.listdir(dataset)

    n_images = len(file_list)
    x_train = np.zeros((n_images, 128, 128, 3))

    for i, file_name in enumerate(file_list):
        if file_name != '.DS_Store':
            image = imread(os.path.join(dataset, file_name))
            x_train[i, :] = (image - 127.5) / 127.5
    print(f"{ConsoleColors.OKGREEN}Done!")

    return x_train
