from posix import listdir
import requests
import tarfile
import os
import numpy as np
from autogoal.datasets import datapath
from tqdm import tqdm

_DATA_PATH = "https://dl.fbaipublicfiles.com/fasttext/data/cooking.stackexchange.tar.gz"
_DATA_DIR = "fasttext_supervised_data"



def download():
    dir_to_save = datapath(_DATA_DIR)
    save_path = datapath("cooking.stackexchange.tar.gz")

    if not os.path.isfile(save_path):
        print("Downloading data for fasttext supervised")
        download_file(_DATA_PATH, save_path)

    if not os.path.isdir(dir_to_save):
        unpack(str(save_path), dir_to_save)

    data_lines  = []
    with open(str(dir_to_save)+"/cooking.stackexchange.txt" ,'r') as f:
        for line in f:
            data_lines.append(line)
        
    _write_file(str(dir_to_save)+"/cooking.train", data_lines[:12404])
    _write_file(str(dir_to_save)+"/cooking.valid", data_lines[12404:])
    

def _write_file(path, lines):
    if not os.path.isfile(path):
        with open(path, 'w')as f:
            f.writelines(lines)





def download_file(url, save_path):
    response = requests.get(url, stream=True)
    with open(save_path, "wb") as file:
        for data in tqdm(response.iter_content()):
            file.write(data)


def unpack(file, dir):
    tar = tarfile.open(file, "r:gz")
    tar.extractall(dir)
    tar.close()


def load():
    download()
    x_train, y_train = _load_dir("/cooking.train")
    x_test, y_test = _load_dir("/cooking.valid")
    return x_train, y_train, x_test, y_test


def _load_dir(dir):
    import re
    from autogoal.experimental.fasttex._base import  Text_Descriptor
    X = []
    y = []
    with open(str(datapath(_DATA_DIR))+dir,"r") as f:
    
        for line in f:
            labels = re.findall("__label__[a-z,A-Z,-]+",line)
            text = re.split("__label__[a-z,A-Z,-]+ ", line)[-1]
            X.append(text)
            y.append(Text_Descriptor( *labels))
    return X, y
