from autogoal.datasets import datapath
import os
import requests
import shutil

IMAGES_URL = "https://www.robots.ox.ac.uk/%7Evgg/data/pets/data/images.tar.gz"

MASKS_URL = "https://www.robots.ox.ac.uk/%7Evgg/data/pets/data/annotations.tar.gz"

IMAGES_DIR = "segmentation_training"
MASKS_DIR = "segmentation_test"


def load():
    download_images()
    download_masks()
    images = load_images()
    masks = load_masks()
    assert len(images) == len(masks)

    test_count = len(images) // 10

    return images[test_count:], masks[test_count:], images[:test_count], masks[:test_count]


def load_images():
    x = []
    for file in os.listdir(datapath(IMAGES_DIR) / "images"):
        if file.endswith(".jpg"):
            x.append(f'{datapath(IMAGES_DIR)}{os.path.sep}{file}')
    return sorted(x)


def load_masks():
    x = []
    for file in os.listdir(datapath(MASKS_DIR) / "annotations" / "trimaps"):
        if file.endswith(".png"):
            x.append(f'{datapath(MASKS_DIR)}{os.path.sep}{file}')
    return sorted(x)


def download_images():
    dir_to_save = datapath(IMAGES_DIR)
    save_path = datapath(IMAGES_DIR + ".tar.gz")

    if not os.path.isfile(save_path):
        print("Downloading training images...")
        download_file(IMAGES_URL, save_path)

    if not os.path.isdir(dir_to_save):
        unpack(str(save_path), dir_to_save)


def download_masks():
    dir_to_save = datapath(MASKS_DIR)
    save_path = datapath(MASKS_DIR + ".tar.gz")

    if not os.path.isfile(save_path):
        print("Downloading test images...")
        download_file(MASKS_URL, save_path)

    if not os.path.isfile(dir_to_save):
        unpack(str(save_path), dir_to_save)


def download_file(url, save_path):
    response = requests.get(url, stream=True)
    with open(save_path, "wb") as file:
        for data in response.iter_content():
            file.write(data)


def unpack(file, dir):
    print(f'Unpacking file {file} in {dir}')
    shutil.unpack_archive(file, dir)
