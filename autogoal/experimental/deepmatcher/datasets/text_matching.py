import numpy as np
from autogoal.datasets import datapath
from tqdm import tqdm

def download():
    save_to_path = datapath("deepmatcher_supervised_data")
    # use tqdm for progress :)

def load():
    download()# download or use local files
    x_train, y_train = -1, -1 # load train
    x_test, y_test = -1, -1 # load test
    return x_train, y_train, x_test, y_test
