from autogoal.kb import Pipeline, Seq, Sentence, Tensor, Dense, Categorical
from autogoal.experimental.pipeline import Supervised
from autogoal.contrib.sklearn import CountVectorizerNoTokenize, CountVectorizer
from autogoal.contrib.nltk._generated import SExprTokenizer, MWETokenizer
from autogoal.contrib.sklearn._generated import ExtraTreeClassifier, SGDClassifier


def test_count_vectorizer_extra_trees_classifier():
    p = Pipeline(
        algorithms=[
            CountVectorizerNoTokenize(
                lowercase=True,
                stopwords_remove=True,
                binary=True,
                inner_tokenizer=SExprTokenizer(strict=True),
                inner_stemmer=SExprTokenizer(strict=True),
                inner_stopwords=MWETokenizer(),
            ),
            ExtraTreeClassifier(
                min_samples_split=2,
                min_weight_fraction_leaf=0.49901406714494645,
                min_impurity_decrease=0.4217179007301969,
                ccp_alpha=0.0,
            ),
        ],
        input_types=(Seq[Sentence], Supervised[Tensor[1, Categorical, Dense]]),
    )

    Xtrain = ["hello world", "this is sparta"]
    ytrain = ["true", "false"]

    p.run(Xtrain, ytrain)


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