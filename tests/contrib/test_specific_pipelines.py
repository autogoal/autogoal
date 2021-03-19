from autogoal.kb import Pipeline, Seq, Sentence, Tensor, Dense, Categorical
from autogoal.contrib.sklearn import CountVectorizerNoTokenize
from autogoal.contrib.nltk._generated import SExprTokenizer, MWETokenizer
from autogoal.contrib.sklearn._generated import ExtraTreeClassifier


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
        input_types=(Seq[Sentence], Tensor[1, Categorical, Dense]),
    )

    Xtrain = ["hello world", "this is sparta"]
    ytrain = ["true", "false"]

    p.run(Xtrain, ytrain)