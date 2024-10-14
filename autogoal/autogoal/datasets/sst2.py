import random
import csv

from autogoal.datasets import download, datapath

def load():
    try:
        download("sst2")
    except Exception as e:
        print(
            "Error loading data. This may be caused due to bad connection. Please delete badly downloaded data and retry"
        )
        raise

    path = datapath("sst2/sst2")

    X_train = []
    y_train = []
    X_test = []
    y_test = []
    with open(path / "train.csv", "r") as fd:
        reader = csv.reader(fd)
        title_line = True
        for row in reader:
            if title_line:
                title_line = False
                continue
            
            X_train.append(row[1])
            y_train.append(int(row[2]))
            
    with open(path / "test.csv", "r") as fd:
        reader = csv.reader(fd)
        title_line = True
        for row in reader:
            if title_line:
                title_line = False
                continue
            
            X_test.append(row[1])
            y_test.append(int(row[2]))
            
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    load()