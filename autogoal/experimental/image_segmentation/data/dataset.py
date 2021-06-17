from autogoal.datasets import datapath
import os
import requests
import shutil

DOWNLOAD_PATH = "https://www.robots.ox.ac.uk/%7Evgg/data/pets/data/images.tar.gz"

TEST_DOWNLOAD_PATH = (
    "https://www.robots.ox.ac.uk/%7Evgg/data/pets/data/annotations.tar.gz"
)

TRAIN_DIR = "segmentation_training"
TEST_DIR = "segmentation_test"


def download_training():
    dir_to_save = datapath(TRAIN_DIR)
    save_path = datapath(TRAIN_DIR + ".tar.gz")

    if not os.path.isfile(save_path):
        print("Downloading training images...")
        download_file(DOWNLOAD_PATH, save_path)

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
    save_path = datapath(TEST_DIR + ".tar.gz")

    if not os.path.isfile(save_path):
        print("Downloading test images...")
        download_file(TEST_DOWNLOAD_PATH, save_path)

    if not os.path.isfile(dir_to_save):
        unpack(str(save_path), dir_to_save)


def load():
    download_training()
    download_test()
