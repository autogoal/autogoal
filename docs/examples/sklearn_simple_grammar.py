# # Integrating with `sklearn`
#
# In this example we wil build a simple grammar based on `sklearn` classifiers
# and apply it to solve a text classification problem
#
# !!! warning
#     This example requires `sklearn` and `nltk` installed, as well as the
#     `"movie_reviews"` corpus from `nltk`. Refer to the documentation on
#     [dependencies](/dependencies/) for further information.

import autogoal.contrib.sklearn # :hide:

# ## Importing the necessary classes
#
# First let's import the relevant classes from `sklearn`.
# We will use three classifiers: support vector machines, logistic regression, and decision trees.

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# For text preprocessing we will use two different strategies: count and tf-idf weighting.

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Optionally, we will try using a singular value decomposition to reduce dimensionality.

from sklearn.decomposition import TruncatedSVD

# Finally, to put it all together, we will use the `sklearn` pipeline API, that allows chaining
# multiple transformers and estimators into a single object on which we can call `fit`.

from sklearn.pipeline import Pipeline as SkPipeline

# Next we import some utilities from `autogoal` that will help us build the grammar.
# These utilities are in the `autogoal.grammar` namespace.

from autogoal.grammar import (
    Continuous,
    Discrete,
    Categorical,
    Union,
    Boolean,
    generate_cfg,
    Sampler,
)

# Next, we will also use two different search strategies, from the `autogoal.search` module.

from autogoal.search import RandomSearch, PESearch
from autogoal.search import ProgressLogger # for logging

# Finally, we will use a toy dataset that comes pre-packaged with `autogoal`.
# This is the famous [Movie Reviews dataset from Pang & Lee](https://www.cs.cornell.edu/people/pabo/movie-review-data/).

from autogoal.datasets import movie_reviews

# ## Wrapping `sklearn` classes
#
# To enable `autogoal`'s automatic grammar inference, we need to provide with annotation
# hints in our classes arguments that describe their types and their possible ranges of values
# Since `sklearn` classes don't come with these annotations, we will wrap its classes into our own.
#
# This also allows us to decide which parameters we actually want to explore with the
# grammar and for which possible values.
# Let's begin with the easier ones, the preprocessing tools.
#
# ### Vectorization
#
# The `CountVectorizer` class has many parameters that we might want to tune, but
# in this example we are interested only in trying different n-gram combinations.
# Hence, we will wrap `CountVectorizer` in our own `Count` class, and redefine its constructor
# to receive an `ngram` parameter. We annotate this parameter with `:Discrete(1,3)` to
# indicate that the possible values are integers in the interval `[1,3]`.
# Of course we also need to call the `super()` initializer and pass the corresponding value.


class Count(CountVectorizer):
    def __init__(self, ngram: Discrete(1, 3)):
        super().__init__(ngram_range=(1, ngram))
        self.ngram = ngram


# !!! note
#     The reason why we store `ngram` in the `__init__()` method is
#     for documentation purposes, so that when we call `print()`
#     we get to see the actual parameters that where selected.
#
# Now we will do the same with the `TfIdfVectorizer` class, but this time we also want to
# explore automatically whether enabling or disabling `use_idf` is better.


class TfIdf(TfidfVectorizer):
    def __init__(self, ngram: Discrete(1, 3), use_idf: Boolean()):
        super().__init__(ngram_range=(1, ngram), use_idf=use_idf)
        self.ngram = ngram
        self.use_idf = use_idf


# ### Dimensionality Reduction
#
# For dimensionality reduction, we want to either use singular value decomposition,
# or nothing at all. The implementation of `TruncatedSVD` is suitable here because it
# provides a fast and scalable approximation to SVDs when dealing with spare matrices.
# As before, we want to parameterize the end dimension, so we will use `:Discrete(50,200)`,
# i.e., if we reduce at all, reduce between `50` and `200` dimensions.


class SVD(TruncatedSVD):
    def __init__(self, n: Discrete(50, 200)):
        super().__init__(n_components=n)
        self.n = n


# To disable dimensionality reduction in some pipelines, it's not correct to simply pass a `None`
# object. That would raise an exception. Instead, we make use of the
# [*Null Object* design pattern](https://en.wikipedia.org/wiki/Null_object_pattern)
# and provide a "no-op" implementation that simply passes through the values.


class NoDec:
    def fit_transform(self, X, y=None):
        return X

    def transform(self, X, y=None):
        return X

    def __repr__(self):
        return "NoDec()"


# !!! note
#     Technically, we could use `"passtrough"` as an argument to the `Pipeline` class that we will
#     use below and achieve the same result. However, this approach is more general and clean, since
#     it doesn't rely on the underlying API providing us with an implementation of the *Null Object* pattern.
#
# ### Classification
#
# Finally, we have to do the same with the classifiers we will use. Since we are already used to this
# code, let's just skim through it.
#
# Quick recap, we have three classifiers, and for each of them we have
# a wrapper class that defines the parameters we actually want to explore, pass them to the underlying
# `sklearn` implementation, and get on with it. In the following code we also use `Categorical`
# and `Continuous` annotations.


class LR(LogisticRegression):
    def __init__(self, penalty: Categorical("l1", "l2"), reg: Continuous(0.1, 10)):
        super().__init__(penalty=penalty, C=reg, solver="liblinear")
        self.penalty = penalty
        self.reg = reg


class SVM(SVC):
    def __init__(
        self, kernel: Categorical("rbf", "linear", "poly"), reg: Continuous(0.1, 10)
    ):
        super().__init__(C=reg, kernel=kernel)
        self.kernel = kernel
        self.reg = reg


class DT(DecisionTreeClassifier):
    def __init__(self, criterion: Categorical("gini", "entropy")):
        super().__init__(criterion=criterion)
        self.criterion = criterion


# ## Creating the `Pipeline`
#
# Now that we have all of the necessary classes with their corresponding parameters
# correctly annotated, it's time to put it all together into a pipeline. We will
# inherit from `sklearn`'s own implementation of `Pipeline`, because we want to fix
# the actual steps that are gonna be used.
#
# Just as before, out initializer declares the parameters. In this case, we
# want a vectorizer, a decomposer and a classifier. To tell `autogoal` to try
# different classes for the same parameter we use the [`Union`](/api/cfg#unions) annotation.
# Likewise, just as before, we have to call the base initializer, this time passing the
# corresponding configuration for an [`sklearn` pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).


class Pipeline(SkPipeline):
    def __init__(
        self,
        vectorizer: Union("Vectorizer", Count, TfIdf),
        decomposer: Union("Decomposer", NoDec, SVD),
        classifier: Union("Classifier", LR, SVM, DT),
    ):
        self.vectorizer = vectorizer
        self.decomposer = decomposer
        self.classifier = classifier

        super().__init__(
            [("vect", vectorizer), ("decomp", decomposer), ("class", classifier),]
        )


# ## Creating the grammar
#
# Once everything is in place, we can tell `autogoal` to automatically infer a grammar
# for all the possible combinations of parameters and clases that we can use.
# The root of our grammar is the `Pipeline` class we just defined. The method `generate_cfg`
# does exactly that, taking a class and building a context free grammar to construct
# that class, based on the parameters' annotations and recursively building the corresponding
# rules for all classes down to basic parameter types.

grammar =  generate_cfg(Pipeline)
print(grammar)

# If you run the code up to this point, it should print something like:
#
# ```bash
# <Pipeline>      := Pipeline (vectorizer=<Vectorizer>, decomposer=<Decomposer>, classifier=<Classifier>)
# <Vectorizer>    := <Count> | <TfIdf>
# <Count>         := Count (ngram=<Count_ngram>)
# <Count_ngram>   := discrete (min=1, max=3)
# <TfIdf>         := TfIdf (ngram=<TfIdf_ngram>, use_idf=<TfIdf_use_idf>)
# <TfIdf_ngram>   := discrete (min=1, max=3)
# <TfIdf_use_idf> := boolean ()
# <Decomposer>    := <NoDec> | <SVD>
# <NoDec>         := NoDec ()
# <SVD>           := SVD (n=<SVD_n>)
# <SVD_n>         := discrete (min=50, max=200)
# <Classifier>    := <LR> | <SVM> | <DT>
# <LR>            := LR (penalty=<LR_penalty>, reg=<LR_reg>)
# <LR_penalty>    := categorical (options=['l1', 'l2'])
# <LR_reg>        := continuous (min=0.1, max=10)
# <SVM>           := SVM (kernel=<SVM_kernel>, reg=<SVM_reg>)
# <SVM_kernel>    := categorical (options=['rbf', 'linear', 'poly'])
# <SVM_reg>       := continuous (min=0.1, max=10)
# <DT>            := DT (criterion=<DT_criterion>)
# <DT_criterion>  := categorical (options=['gini', 'entropy'])
# ```
#
# Notice how the grammar specifies all the possible ways to build a `Pipeline`,
# both considering the different implementations we have for vectorizers, decomposers and classifiers;
# as well as their corresponding parameters. Our grammar is fairly simple because we only have
# two levels of recursion, Pipeline and its parameters; but this same process can be applied to any
# hierarchy of any complexity, including circular references.
#
# Let's take a look at how different pipelines can be generated with this grammar by sampling
# 10 random pipelines.

for _ in range(10):
    print(grammar.sample())

# You should see something like this, but your exact pipelines will be different due to random sampling.
#
# ```bash
# Pipeline(classifier=LR(penalty='l2', reg=2.381707354390544),
#          decomposer=SVD(n=99), vectorizer=TfIdf(ngram=2, use_idf=True))
# Pipeline(classifier=DT(criterion='gini'), decomposer=NoDec(),
#          vectorizer=TfIdf(ngram=1, use_idf=True))
# Pipeline(classifier=LR(penalty='l2', reg=6.910387345035382), decomposer=NoDec(),
#          vectorizer=Count(ngram=1))
# Pipeline(classifier=DT(criterion='gini'), decomposer=SVD(n=147),
#          vectorizer=Count(ngram=2))
# Pipeline(classifier=DT(criterion='entropy'), decomposer=SVD(n=97),
#          vectorizer=TfIdf(ngram=1, use_idf=True))
# Pipeline(classifier=LR(penalty='l1', reg=5.673920890471801),
#          decomposer=SVD(n=98), vectorizer=TfIdf(ngram=1, use_idf=False))
# Pipeline(classifier=SVM(kernel='poly', reg=4.658013333072327),
#          decomposer=NoDec(), vectorizer=Count(ngram=1))
# Pipeline(classifier=SVM(kernel='linear', reg=7.105445268768045),
#          decomposer=NoDec(), vectorizer=TfIdf(ngram=2, use_idf=False))
# Pipeline(classifier=SVM(kernel='linear', reg=1.7737620816556527),
#          decomposer=NoDec(), vectorizer=TfIdf(ngram=3, use_idf=True))
# Pipeline(classifier=DT(criterion='gini'), decomposer=NoDec(),
#          vectorizer=Count(ngram=2))
# ```
#
# ## Finding the best pipeline
#
# To continue with the example, we will now search for the best pipeline.
# We will explore two different search strategies: a random search and a probabilistic
# evolutionary search that _should_ be better than the random.
#
# We will evaluate our pipelines on the `movie_reviews` corpus. For that purpose
# we need a fitness function, which is a simple callable that takes a pipeline and outputs
# a score. Fortunately, the `movie_reviews.make_fn` function does this for us, taking
# care of train/test splitting, fitting a pipeline in the training set and computing
# the accuracy on the test set.

fitness_fn = movie_reviews.make_fn(examples=100)

# ### Random search
#
# The `RandomSearch` strategy simply calls `grammar.sample()` a bunch of times
# and stores the best performing pipeline. It has no intelligence whatsoever,
# but it serves as a good baseline implementation.
#
# We will run it for a total of `1000` fitness evaluations, or equivalently, a total
# of `1000` different random pipelines. To see what's actually going on we will use
# the wonderfull `enlighten` library through our implementation `EnlightenLogger`.

logger = ProgressLogger(log_solutions=True)

random_search = RandomSearch(grammar, fitness_fn, random_state=0)
best_rand, fn_rand = random_search.run(1000, logger=logger)

# !!! note
#     For reproducibility purposes we can pass a fixed random seed in `random_state`.
#
# ### Evolutionary Search
#
# Random search is fun, but to search with purpose, we need a more intelligent sampling
# strategy. The `PESearch` (short for Probabilistic Evolutionary Search, phew), does just that.
# It starts with a random sampling strategy, but as it evaluates more pipelines, it modifies
# an probabilistic sampling model so that pipelines similar to the best ones found are more
# commonly sampled.
#
# There are three main parameters for `PESearch`.
#
# * The `pop_size` parameter indicates how many pipelines
# to sample between each update to the probabilistic model.
# * The `selection` parameter indicates how
# many of the best pipelines found are used to update the model.
# * The `learning_factor` allows to weight more the new updated model against the previous one.
#
# All in all these three parameters control the balance between **exploration** and **exploitation**,
# which is a complicated topic, to say the least, in evolutionary optimization.
# Here we set them to sensible values, but when in doubt, just use the defaults.

pge = PESearch(grammar, fitness_fn, pop_size=10, selection=0.2, learning_factor=0.1, random_state=0)
best_pge, fn_pge = pge.run(1000, logger=logger)

# Finally let's see what came through:

print("PESearch     :", fn_pge, "\n", best_pge)
print("RandomSearch :", fn_rand, "\n", best_rand)

# When both searches are over, you'll see something like:
#
# ```bash
# PESearch     : 0.8
# Pipeline(classifier=DT(criterion='gini'), decomposer=SVD(n=62),
#          vectorizer=TfIdf(ngram=1, use_idf=True))
# RandomSearch : 0.8
# Pipeline(classifier=DT(criterion='gini'), decomposer=SVD(n=198),
#          vectorizer=Count(ngram=2))
# ```
