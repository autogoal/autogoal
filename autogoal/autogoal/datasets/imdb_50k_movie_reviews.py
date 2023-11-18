import random

from autogoal.datasets import download, datapath


def load(max_examples=None):
    try:
        download("imdb_50k_movie_reviews")
    except:
        print(
            "Error loading data. This may be caused due to bad connection. Please delete badly downloaded data and retry"
        )
        raise

    sentences = []
    classes = []

    path = datapath("imdb_50k_movie_reviews")
    
    sentences = []
    classes = []
    with open(path / "imdb_50k_movie_reviews" / "imdb_50k_movie_reviews.csv", "r") as fd:
        title_line = True
        for i in fd.readlines():
            if title_line:
                title_line = False
                continue

            if max_examples and len(sentences) >= max_examples:
                break

            clean_line = i.strip().split(",")

            sentences.append(clean_line[0])
            classes.append(clean_line[-1])

    return sentences, classes


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