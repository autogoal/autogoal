import random

from autogoal.datasets import download, datapath


def load(max_examples=None):
    try:
        download("movie_reviews")
    except:
        print(
            "Error loading data. This may be caused due to bad connection. Please delete badly downloaded data and retry"
        )
        raise

    sentences = []
    classes = []

    path = datapath("movie_reviews")

    ids = list(path.rglob("*.txt"))
    random.shuffle(ids)

    for fd in ids:
        if "neg/" in str(fd):
            cls = "neg"
        else:
            cls = "pos"

        with fd.open() as fp:
            sentences.append(fp.read())
            classes.append(cls)

        if max_examples and len(classes) >= max_examples:
            break

    return sentences, classes


def make_fn(test_size=0.25, examples=None):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X, y = load(examples)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    def fitness_fn(pipeline):
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        return accuracy_score(y_test, y_pred)

    return fitness_fn
