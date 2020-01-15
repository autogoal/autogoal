import random


def load(max_examples=None):
    from nltk.corpus import movie_reviews

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
    from sklearn.metrics import accuracy_score

    X, y = load(examples)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    def fitness_fn(pipeline):
        pipeline.send("train")
        pipeline.run((X_train, y_train))
        pipeline.send("eval")
        _, y_pred = pipeline.run((X_test, [None] * len(y_test)))

        return accuracy_score(y_test, y_pred)

    return fitness_fn
