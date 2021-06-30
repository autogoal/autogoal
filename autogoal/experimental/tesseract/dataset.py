from os import listdir
import pandas as pd
from autogoal import datasets
from autogoal.datasets import download_and_save, datapath, unpack

data_url = "https://github.com/dayfundora/pytesseract-dataset/raw/main/pytesseract-dataset.zip"
dataset = "pytesseract-dataset"

def load():
    # Download the data
    fname = f"{dataset}.zip"
    file_path = datapath(fname)
    if not file_path.exists():
        print(f"Downloading {dataset}...")
        download_and_save(data_url, file_path, True)
        print("Done!\n")

    # Unpack the zip file
    dir_path = datapath(dataset)
    if not dir_path.exists():
        print(f"Unpacking {fname}...")
        unpack(fname)
        print("Done.\n")

    # build the training and test sets
    data = pd.read_csv(datapath(dataset) / "dataset.csv")
    X_unopen=list(data['image'])
    y_unopen=list(data['text'])
    
    X=[f"{dir_path}/{image_name}" for image_name in X_unopen]
    y=[open(f"{dir_path}/{text_dir}",'r',encoding='ISO-8859-1').read() for text_dir in y_unopen]

    length=len(y)
    train_test_index=length//5
    
        
    X_train = X[:-train_test_index]
    y_train = y[:-train_test_index]
    X_test = X[-train_test_index:]
    y_test = y[-train_test_index:]
    return X_train, y_train, X_test, y_test