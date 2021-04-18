from autogoal.contrib.sklearn import CountVectorizer
from autogoal.contrib.sklearn._generated import SGDClassifier
from autogoal.contrib.nltk._generated import ClassifierBasedPOSTagger
from autogoal.kb import Supervised
from autogoal.kb import Categorical, Dense, Pipeline, Sentence, Seq, Tensor, Word, Postag


def test_count_vectorizer_sgd():
    p = Pipeline(
        algorithms=[
            CountVectorizer(lowercase=True, binary=True),
            SGDClassifier(
                loss="perceptron",
                penalty="l1",
                l1_ratio=0.999,
                fit_intercept=False,
                tol=0.001,
                shuffle=True,
                epsilon=0.24792790326293826,
                learning_rate="optimal",
                eta0=0.992,
                power_t=4.991,
                early_stopping=False,
                validation_fraction=0.993,
                n_iter_no_change=1,
                average=True,
            ),
        ],
        input_types=(Seq[Sentence], Supervised[Tensor[1, Categorical, Dense]]),
    )

    Xtrain = ["hello world", "this is sparta"]
    ytrain = ["true", "false"]

    p.run(Xtrain, ytrain)
    p.send("eval")
    p.run(Xtrain, None)


def test_classifier_tagger():
    p = Pipeline(
        algorithms=[
            ClassifierBasedPOSTagger(),
        ],
        input_types=(Seq[Seq[Word]], Supervised[Seq[Seq[Postag]]]),
    )

    p.run(
        [["hello", "world"]],
        [["A", "B"]]
    )

    p.send("eval")

    result = p.run([["hello", "world"]], None)
    assert result == [["A", "B"]]
