
# Integrating with `sklearn`

In this example we wil build a simple grammar based on `sklearn` classifiers
and apply it to solve a text classification problem

!!! warning
    This example requires `sklearn` and `nltk` installed, as well as the
    `"movie_reviews"` corpus from `nltk`. Refer to the documentation on
    [dependencies](/dependencies/) for further information.

## Importing the necessary classes

First let's import the relevant classes from `sklearn`.
We will use three classifiers: support vector machines, logistic regression, and decision trees.

```python

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

```

For text preprocessing we will use two different strategies: count and tf-idf weighting.

```python

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

```

Optionally, we will try using a singular value decomposition to reduce dimensionality.

```python

from sklearn.decomposition import TruncatedSVD

```

Finally, to put it all together, we will use the `sklearn` pipeline API, that allows chaining
multiple transformers and estimators into a single object on which we can call `fit`.

```python

from sklearn.pipeline import Pipeline as SkPipeline

```

Next we import some utilities from `autogoal` that will help us build the grammar.
These utilities are in the `autogoal.grammar` namespace.

```python

from autogoal.grammar import (
    Continuous,
    Discrete,
    Categorical,
    Union,
    Boolean,
    generate_cfg,
)

```

Next, we will also use two different search strategies, from the `autogoal.search` module.

```python

from autogoal.search import RandomSearch, PESearch

```

Finally, we will use a toy dataset that comes pre-packaged with `autogoal`.
This is the famous [Movie Reviews dataset from Pang & Lee](https://www.cs.cornell.edu/people/pabo/movie-review-data/).

```python

from autogoal.datasets import movie_reviews

```

## Wrapping `sklearn` classes

To enable `autogoal`'s automatic grammar inference, we need to provide with annotation
hints in our classes arguments that describe their types and their possible ranges of values
Since `sklearn` classes don't come with these annotations, we will wrap its classes into our own.

This also allows us to decide which parameters we actually want to explore with the
grammar and for which possible values.
Let's begin with the easier ones, the preprocessing tools.

### Preprocessing

The `CountVectorizer` class has many parameters that we might want to tune, but
in this example we are interested only in trying different n-gram combinations.
Hence, we will wrap `CountVectorizer` in our own `Count` class, and redefine its constructor
to receive an `ngram` parameter. We annotate this parameter with `:Discrete(1,3)` to
indicate that the possible values are integers in the interval `[1,3]`.
Of course we also need to call the `super()` initializer and pass the corresponding value.

```python


class Count(CountVectorizer):
    def __init__(self, ngram: Discrete(1, 3)):
        super(Count, self).__init__(ngram_range=(1, ngram))
        self.ngram = ngram

    def __repr__(self):
        return "Count(ngram=%r)" % self.ngram


```

!!! note
    The `__repr__()` method is only here for documentation purposes, so that when we call `print()`
    we get to see the actual parameters that where selected. That's also the reason why we store
    `ngram` in the `__init__()` method.

Now we will do the same with the `TfIdfVectorizer` class, but this time we also want to
explore automatically whether enabling or disabling `use_idf` is better.

```python


class TfIdf(TfidfVectorizer):
    def __init__(self, ngram: Discrete(1, 3), use_idf: Boolean()):
        super(TfIdf, self).__init__(ngram_range=(1, ngram), use_idf=use_idf)
        self.ngram = ngram
        self.use_idf = use_idf

    def __repr__(self):
        return "TfIdf(ngram=%r, use_idf=%r)" % (self.ngram, self.use_idf)


```

### Dimensionality Reduction

For dimensionality reduction, we want to either use singular value decomposition,
or nothing at all. The implementation of `TruncatedSVD` is suitable here because it
provides a fast and scalable approximation to SVDs when dealing with spare matrices.
As before, we want to parameterize the end dimension, so we will use `:Discrete(50,200)`,
i.e., if we reduce at all, reduce between `50` and `200` dimensions.

```python


class SVD(TruncatedSVD):
    def __init__(self, n: Discrete(50, 200)):
        super(SVD, self).__init__(n_components=n)
        self.n = n

    def __repr__(self):
        return "SVD(n=%r)" % self.n


```

To disable dimensionality reduction in some pipelines, it's not correct to simply pass a `None`
object. That would raise an exception. Instead, we make use of the
[*Null Object* design pattern](https://en.wikipedia.org/wiki/Null_object_pattern)
and provide a "no-op" implementation that simply passes through the values.

```python


class NoDec:
    def fit_transform(self, X, y=None):
        return X

    def transform(self, X, y=None):
        return X

    def __repr__(self):
        return "NoDec()"


```

!!! note
    Technically, we could use `"passtrough"` as an argument to the `Pipeline` class that we will
    use below and achieve the same result. However, this approach is more general and clean, since
    it doesn't rely on the underlying API providing us with an implementation of the *Null Object* pattern.

### Classification

Finally, we have to do the same with the classifiers we will use. Since we are already used to this
code, let's just skim through it.

Quick recap, we have three classifiers, and for each of them we have
a wrapper class that defines the parameters we actually want to explore, pass them to the underlying
`sklearn` implementation, and get on with it. In the following code we also use `Categorical`
and `Continuous` annotations.

```python


class LR(LogisticRegression):
    def __init__(self, penalty: Categorical("l1", "l2"), reg: Continuous(0.1, 10)):
        super(LR, self).__init__(penalty=penalty, C=reg, solver="liblinear")
        self.penalty = penalty
        self.reg = reg

    def __repr__(self):
        return "LR(penalty=%r, reg=%r)" % (self.penalty, self.reg)


class SVM(SVC):
    def __init__(
        self, kernel: Categorical("rbf", "linear", "poly"), reg: Continuous(0.1, 10)
    ):
        super(SVM, self).__init__(C=reg, kernel=kernel)
        self.kernel = kernel
        self.reg = reg

    def __repr__(self):
        return "SVM(kernel=%r, reg=%r)" % (self.kernel, self.reg)


class DT(DecisionTreeClassifier):
    def __init__(self, criterion: Categorical("gini", "entropy")):
        super(DT, self).__init__(criterion=criterion)
        self.criterion = criterion

    def __repr__(self):
        return "DT(criterion=%r)" % self.criterion


```

## Creating the `Pipeline`

Now that we have all of the necessary classes with their corresponding parameters
correctly annotated, it's time to put it all together into a pipeline. We will
inherit from `sklearn`'s own implementation of `Pipeline`, because we want to fix
the actual steps that are gonna be used.

Just as before, out initializer declares the parameters. In this case, we
want a vectorizer, a decomposer and a classifier. To tell `autogoal` to try
different classes for the same parameter we use the [`Union`](/api/cfg#unions) annotation. #

```python

class Pipeline(SkPipeline):
    def __init__(
        self,
        vectorizer: Union("Vectorizer", Count, TfIdf),
        decomposer: Union("Decomposer", NoDec, SVD),
        classifier: Union("Classifier", LR, SVM, DT),
    ):
        super(Pipeline, self).__init__(
            [
                ("vect", self.vectorizer),
                ("decomp", self.decomposer),
                ("class", self.classifier),
            ]
        )


```


## Creating the grammar


```python


def main():
    grammar = generate_cfg(Pipeline)
    print(grammar)

    pge = PESearch(
        grammar, movie_reviews.make_fn(max_examples=100), pop_size=10, random_state=0
    )
    best_pge, fn_pge = pge.run(1000)

    random_search = RandomSearch(
        grammar, movie_reviews.make_fn(max_examples=100), random_state=0
    )
    best_rand, fn_rand = random_search.run(1000)

    print("Best with PGE", fn_pge, "vs Random", fn_rand)


if __name__ == "__main__":
    main()
```
