from autogoal.datasets import datapath
import os
import requests
import shutil
from posix import listdir
import numpy as np


IMAGES_PATH = "https://www.robots.ox.ac.uk/%7Evgg/data/pets/data/images.tar.gz"
TEST_PATH = (
    "https://www.robots.ox.ac.uk/%7Evgg/data/pets/data/annotations.tar.gz"
)

TRAIN_DIR="segmentation_training"
TEST_DIR="segmentation_test"


def download_training():
    dir_to_save = datapath(IMAGES_PATH)
    save_path = datapath(TRAIN_DIR+".tar.gz")

    if not os.path.isfile(save_path):
        download_file(IMAGES_PATH, save_path)

    if not os.path.isdir(dir_to_save):
        unpack(str(save_path), dir_to_save)
        
def download_file(url, save_path):
    response = requests.get(url, stream=True)
    with open(save_path, "wb") as file:
        for data in response.iter_content():
            file.write(data)
            
def unpack(file, dir):
    shutil.unpack_archive(file, dir)



def download_test():
    dir_to_save = datapath(TEST_DIR)
    save_path = datapath(TEST_DIR+".tar.gz")

    if not os.path.isfile(save_path):
        print("Downloading test samples for audio commands dataset.")
        download_file(TEST_PATH, save_path)

    if not os.path.isfile(dir_to_save):
        unpack(str(save_path), dir_to_save)


def load():
    download_training()
    download_test()
    x_train, y_train = _load(datapath(TRAIN_DIR))
    x_test, y_test = _load(datapath(TEST_DIR))
    return x_train, y_train, x_test, y_test
