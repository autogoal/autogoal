import random
from nltk.corpus import movie_reviews


def load(max_examples=None):
    sentences = []
    classes = []

    ids = list(movie_reviews.fileids())
    random.shuffle(ids)

    for fd in ids:
        if fd.startswith("neg/"):
            cls = "neg"
        else:
            cls = "pos"

        fp = movie_reviews.open(fd)
        sentences.append(fp.read())
        classes.append(cls)

        if max_examples and len(classes) >= max_examples:
            break

    return sentences, classes


def make_fn(test_size=0.25, examples=None):
    from sklearn.model_selection import train_test_split

    X, y = load(examples)
    X_train, X_test, y_train, y_test = _split(X, y, test_size=test_size)

    def fitness_fn(pipeline):
        pipeline.fit(X_train, y_train)
        return pipeline.score(X_test, y_test)

    return fitness_fn
