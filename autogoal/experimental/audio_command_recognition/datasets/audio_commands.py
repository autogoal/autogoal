from posix import listdir
import requests
import shutil
import os
import numpy as np
from autogoal.datasets import datapath
from tqdm import tqdm

_DOWNLOAD_PATH = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
_TEST_DOWNLOAD_PATH_ = (
    "http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz"
)
_TRAINING_DIR = "audio_command_training"
_TEST_DIR = "audio_command_test"
_LABELS = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
    "_silence_",
    "_unknown_",
]


def download_training():
    dir_to_save = datapath(_TRAINING_DIR)
    save_path = datapath("audio_command_training.tar.gz")

    if not os.path.isfile(save_path):
        print("Downloading training samples for audio commands dataset.")
        download_file(_DOWNLOAD_PATH, save_path)

    if not os.path.isdir(dir_to_save):
        unpack(str(save_path), dir_to_save)


def download_test():
    dir_to_save = datapath(_TEST_DIR)
    save_path = datapath("audio_command_test.tar.gz")

    if not os.path.isfile(save_path):
        print("Downloading test samples for audio commands dataset.")
        download_file(_TEST_DOWNLOAD_PATH_, save_path)

    if not os.path.isfile(dir_to_save):
        unpack(str(save_path), dir_to_save)


def download_file(url, save_path):
    response = requests.get(url, stream=True)
    with open(save_path, "wb") as file:
        for data in tqdm(response.iter_content()):
            file.write(data)


def unpack(file, dir):
    shutil.unpack_archive(file, dir)


def load():
    download_training()
    download_test()
    x_train, y_train = _load_dir(datapath(_TRAINING_DIR))
    x_test, y_test = _load_dir(datapath(_TEST_DIR))
    return x_train, y_train, x_test, y_test


def _load_dir(dir):
    x, y = [], []
    for label in _LABELS:
        audio_dir = f"{dir}{os.path.sep}{label}"
        for file in listdir(audio_dir):
            if file.endswith(".wav"):
                x.append(f"{audio_dir}{os.path.sep}{file}")
                y.append(label)
    return x, np.array(y)

