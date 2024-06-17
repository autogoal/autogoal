import random
import csv

from autogoal.datasets import download, datapath

def load(onehot = False):
    try:
        download("imdb_50k_movie_reviews")
    except:
        print(
            "Error loading data. This may be caused due to bad connection. Please delete badly downloaded data and retry"
        )
        raise

    path = datapath("imdb_50k_movie_reviews")
    
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
            
            X_train.append(row[0])
            y_train.append(row[1])
            
    with open(path / "test.csv", "r") as fd:
        reader = csv.reader(fd)
        title_line = True
        for row in reader:
            if title_line:
                title_line = False
                continue
            
            X_test.append(row[0])
            y_test.append(row[1])
            
    if (onehot):
        X_train, y_train = _load_onehot(X_train, y_train)
        X_test, y_test = _load_onehot(X_test, y_test)
    return X_train, y_train, X_test, y_test

def make_fn(test_size=0.5, examples=None):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X, y = load(examples)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    def fitness_fn(pipeline):
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        return accuracy_score(y_test, y_pred)

    return fitness_fn

def _load_onehot(X, Y):
    return X, [0 if y == "negative" else 1 for y in Y]

if __name__ == "__main__":
    load()