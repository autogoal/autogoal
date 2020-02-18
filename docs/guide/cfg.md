# Class-based API

AutoGOAL's class-based API allows to automatically find optimal instances of complex objects
in user-defined class hierarchies that solve a given task. A task is simply some method that
evaluates an object's performance. The solution space is defined by a class hierarchy and
all possible ways of combining instances of different types, and creating them with different parameters.

!!! note
    The following code requires `sklearn` dependencies. Read the [dependencies section](/dependencies/) for more information.

For example, suppose we want to build the best possible classifier in `scikit-learn` for a given dataset.
Let's begin with a simple classification problem.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
```

We use `make_classification` to create a toy classification problem.

```python
X, y = make_classification(random_state=0)  # Fixed seed for reproducibility
```

One first idea is to use a specific algorithm, such as Logistic Regression, to solve this problem.
Since the nature of these problems is stochastic, we need to train in one subset, test on another,
and perform a sensible number of evaluations to actually know if this is any good.

```python
from sklearn.linear_model import LogisticRegression


def evaluate(estimator, iters=30):
    scores = []

    for i in range(iters):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        estimator.fit(X_train, y_train)
        scores.append(estimator.score(X_test, y_test))

    return sum(scores) / len(scores)


lr = LogisticRegression()
score = evaluate(lr)  # around 0.83
```

So far so good, but maybe we could do better with a different set of parameters.
Logistic regression has at least two parameters that influence heavily its performance: the penalty function
and the regularization strength.

Instead of writing a loop through a bunch of different parameters, we can use AutoGOAL to automatically
explore the space of possible combinations.
We can do this with the class-based API by providing annotations for the parameters we want to explore.

## Trying multiple logistic regressions

First we import some annotation types from AutoGOAL

```python
from autogoal.grammar import Continuous, Categorical
```

Next we annotate the parameters we want to explore.
Since we cannot modify the class `LogisticRegression` we will inherit from it.

```python
class LR(LogisticRegression):
    def __init__(self, penalty: Categorical("l1", "l2"), C: Continuous(0.1, 10)):
        super().__init__(penalty=penalty, C=C, solver="liblinear")
```

The `penalty: Categorical("l1", "l2")` annotation tells AutoGOAL that for this class the
parameter `penalty` can take values from a list of predefined values. Likewise the
`C: Continuous(0.1, 10)` annotation indicates that the parameter `C` can take a float value in a specified range.

Now we will use AutoGOAL to automatically generate different instances of our `LR` class.
With the class-based API we achieve this by building a **context-free grammar** that describes all possible instances.

```python
from autogoal.grammar import  generate_cfg

grammar =  generate_cfg(LR)
print(grammar)
```

```bash
<LR>         := LR (penalty=<LR_penalty>, C=<LR_C>)
<LR_penalty> := categorical (options=['l1', 'l2'])
<LR_C>       := continuous (min=0.1, max=10)
```

As you can see, this grammar describes the set of all possible instances of `LR` by
describing how to call the constructor, and how to generate random values for its parameters.

!!! note
    Formally, this a called a **Context-free grammar**. They are used in Computer Science to describe formal languages,
    such as programming languages, mathematical expresions, etc.
    Context-free grammars work by describing a set of replacement rules that you can apply recursively to
    construct a string of a specific language. In this case we are using grammars to describe the language
    of all possible Python codes that instantiates an `LR`.
    You can read more in [Wikipedia](https://en.wikipedia.org/wiki/Context-free_grammar).

You can use this grammar to generate a bunch of random instances.

```python
for _ in range(5):
    print(grammar.sample())
```

```bash
LR(C=4.015231900472649, penalty='l2')
LR(C=9.556786605505499, penalty='l2')
LR(C=4.05716261883461, penalty='l1')
LR(C=3.2786487445120858, penalty='l1')
LR(C=4.655510386502897, penalty='l2')
```

Now we can search for the best combination of constructor parameters by
trying a bunch of different instances and see which one obtains the best score.
AutoGOAL also has tools for automating this process.

```python
from autogoal.search import RandomSearch

search = RandomSearch(grammar, evaluate, random_state=0)  # Fixed seed
best, score = search.run(100)

print("Best:", best, "\nScore:", score)
```

The `RandomSearch` will try 100 different random instances, and for each one
run the `evaluate` method we defined earlier. It returns the best one and the corresponding score.

```
Best: LR(C=0.7043201482743121, penalty='l1')
Score: 0.8853333333333337
```

So we can do a little bit better by carefully selecting the right parameters.
However, maybe we can do even better.

## Trying different algorithms

To continue this line of thought, maybe we could do better with a different classifier.
We could try decision trees,
support vector machines, naive bayes, and many more.
Here is the first time AutoGOAL can come to our aid. Instead of writing ourselves
a loop through all the possible classes, we can do the following.

First, we import everything we need.

```python
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
```

Now that we have all the classes we want to try, we have to tell AutoGOAL that there is something
to optimize.
We start by defining a space of possible parameters that we want to tune for each of these
classes. Like with `LR`, we will wrap these classes in our own to provide the corresponding annotations.

```python
class SVM(SVC):
    def __init__(
        self, kernel: Categorical("rbf", "linear", "poly"), C: Continuous(0.1, 10)
    ):
        super().__init__(C=C, kernel=kernel)


class DT(DecisionTreeClassifier):
    def __init__(self, criterion: Categorical("gini", "entropy")):
        super().__init__(criterion=criterion)


class NB(GaussianNB):
    def __init__(self, var_smoothing: Continuous(1e-10, 0.1)):
        super().__init__(var_smoothing=var_smoothing)
```

Next, we use AutoGOAL to construct a grammar for the union of the possible instances
of each of these clases.

```python
from autogoal.grammar import Union
from autogoal.grammar import generate_cfg

grammar =  generate_cfg(Union("Classifier", LR, SVM, NB, DT))
```

!!! note
    The method [`generate_cfg`](/api/grammar/#generate_cfg) works not only with annotated classes
    but also with plain methods, or anything that has a `__call__` and suitable annotations.

This grammar defines all possible ways to obtain a `Classifier`, which is basically
by instantiating one of the classes we gave it with a suitable value for each parameter.
We can test it by generating a few of them.

```python
print(grammar)
```

```bash
<Classifier>       := <LR> | <SVM> | <NB> | <DT>
<LR>               := LR (penalty=<LR_penalty>, C=<LR_C>)
<LR_penalty>       := categorical (options=['l1', 'l2'])
<LR_C>             := continuous (min=0.1, max=10)
<SVM>              := SVM (kernel=<SVM_kernel>, C=<SVM_C>)
<SVM_kernel>       := categorical (options=['rbf', 'linear', 'poly'])
<SVM_C>            := continuous (min=0.1, max=10)
<NB>               := NB (var_smoothing=<NB_var_smoothing>)
<NB_var_smoothing> := continuous (min=1e-10, max=0.1)
<DT>               := DT (criterion=<DT_criterion>)
<DT_criterion>     := categorical (options=['gini', 'entropy'])
```

!!! note
    The constructor for `Union` requires as first parameter a `name` so that in the grammar
    a suitable production can be defined. Think of it as the name of an abstract class that
    groups all your classes, just there is no actual type ever created, it's just for organizational purposes.

```python
for _ in range(5):
    print(grammar.sample())
```

```bash
NB(var_smoothing=0.04620465447733762)
DT(criterion='gini')
SVM(C=3.2914771222720116, kernel='rbf')
LR(C=7.809744923904822, penalty='l1')
DT(criterion='gini')
```

Now that we have a bunch of possible algorithms, let's see which one is best.

```python
search = RandomSearch(grammar, evaluate, random_state=0)
best, score = search.run(100)

print("Best:", best, "\nScore:", score)
```

```bash
Best: NB(var_smoothing=0.08450775758264377)
Score: 0.8840000000000003
```

So it doesn't really seem that we can do much better, which is unsurprising given that we are only
doing a random search (there are [better search methods](/api/search/) in AutoGOAL), and this
is a toy problem which basically any algorithm can solve fairly well.

However, to continue with the example, now that we know how to optimize any given grammar,
what is interesting is can we increase the complexity of our pipeline by adding more
and more layers and steps to it, to solve more challenging problems.

## Adding more steps

To illustrate how to build more complex pipelines, let's change our focus to a bit more challenging problem:
[sentiment analysis](https://en.wikipedia.org/wiki/Sentiment_analysis).
We will use the ultra-know [movie reviews corpus](https://www.cs.cornell.edu/people/pabo/movie-review-data/)
as a testbed in the next few examples.

```python
from autogoal.datasets import movie_reviews
```

To solve sentiment analysis we need to add a step before the actual classification in order to get
feature matrices from text. The simplest solution is to use a vectorizer from `scikit-learn`.
There are two options to choose from.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
```

The `CountVectorizer` class has many parameters that we might want to tune, but
in this example we are interested only in trying different n-gram combinations.
Hence, we will wrap `CountVectorizer` in our own `Count` class, and redefine its constructor
to receive an `ngram` parameter. We annotate this parameter with `:Discrete(1,3)` to
indicate that the possible values are integers in the interval `[1,3]`.

```python
from autogoal.grammar import Discrete


class Count(CountVectorizer):
    def __init__(self, ngram: Discrete(1, 3)):
        super().__init__(ngram_range=(1, ngram))
        self.ngram = ngram
```

!!! note
    The reason why we store `ngram` in the `__init__()` method is
    for documentation purposes, so that when we call `print()`
    we get to see the actual parameters that where selected.
    This works automatically for parameters that are named exactly as `sklearn`
    parameters, because their `__repr__` takes care, but for parameters which we
    introduce we need to store them in the instance so that `__repr__` works.

Now we will do the same with the `TfIdfVectorizer` class, but this time we also want to
explore automatically whether enabling or disabling `use_idf` is better.
We will use the `Boolean` annotation in this case.

```python
from autogoal.grammar import Boolean


class TfIdf(TfidfVectorizer):
    def __init__(self, ngram: Discrete(1, 3), use_idf: Boolean()):
        super().__init__(ngram_range=(1, ngram), use_idf=use_idf)
        self.ngram = ngram
```

Besides vectorization, another common step in NLP pipelines is dimensionality reduction.
For dimensionality reduction, we want to either use singular value decomposition,
or nothing at all. The implementation of `TruncatedSVD` is suitable here because it
provides a fast and scalable approximation to SVDs when dealing with spare matrices.
As before, we want to parameterize the end dimension, so we will use `:Discrete(50,200)`,
i.e., if we reduce at all, reduce between `50` and `200` dimensions.
We will use the `Discrete` annotation in this case.

```python
from sklearn.decomposition import TruncatedSVD


class SVD(TruncatedSVD):
    def __init__(self, n: Discrete(50, 200)):
        super().__init__(n_components=n)
        self.n = n
```

To disable dimensionality reduction in some pipelines, it's not correct to simply pass a `None`
object. That would raise an exception. Instead, we make use of the
[*Null Object* design pattern](https://en.wikipedia.org/wiki/Null_object_pattern)
and provide a "no-op" implementation that simply passes through the values.

```python
class Noop:
    def fit_transform(self, X, y=None):
        return X

    def transform(self, X, y=None):
        return X

    def __repr__(self):
        return "Noop()"
```

!!! note
    Technically, we could use `"passtrough"` as an argument to the `Pipeline` class that we will
    use below and achieve the same result. However, this approach is more general and clean, since
    it doesn't rely on the underlying API providing us with an implementation of the *Null Object* pattern.

Now that we have all of the necessary classes with their corresponding parameters
correctly annotated, it's time to put it all together into a pipeline. We will
inherit from `sklearn`'s own implementation of `Pipeline`, because we want to fix
the actual steps that are gonna be used.

Just as before, out initializer declares the parameters. In this case, we
want a vectorizer, a decomposer and a classifier. To tell `autogoal` to try
different classes for the same parameter we use the [`Union`](/api/cfg#unions) annotation.
Likewise, just as before, we have to call the base initializer, this time passing the
corresponding configuration for an
[`sklearn` pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).

```python
from sklearn.pipeline import Pipeline as _Pipeline


class Pipeline(_Pipeline):
    def __init__(
        self,
        vectorizer: Union("Vectorizer", Count, TfIdf),
        decomposer: Union("Decomposer", Noop, SVD),
        classifier: Union("Classifier", LR, SVM, DT, NB),
    ):
        self.vectorizer = vectorizer
        self.decomposer = decomposer
        self.classifier = classifier

        super().__init__(
            [
                ("vec", vectorizer),
                ("dec", decomposer),
                ("cls", classifier),
            ]
        )
```

Once everything is in place, we can tell `autogoal` to automatically infer a grammar
for all the possible combinations of parameters and clases that we can use.
The root of our grammar is the `Pipeline` class we just defined. The method `generate_cfg`
does exactly that, taking a class and building a context free grammar to construct
that class, based on the parameters' annotations and recursively building the corresponding
rules for all classes down to basic parameter types.

```python
grammar =  generate_cfg(Pipeline)
```

Notice how the grammar specifies all the possible ways to build a `Pipeline`,
both considering the different implementations we have for vectorizers, decomposers and classifiers;
as well as their corresponding parameters. Our grammar is fairly simple because we only have
two levels of recursion, Pipeline and its parameters; but this same process can be applied to any
hierarchy of any complexity, including circular references.

```python
print(grammar)
```

```bash
<Pipeline>         := Pipeline (vectorizer=<Vectorizer>, decomposer=<Decomposer>, classifier=<Classifier>)
<Vectorizer>       := <Count> | <TfIdf>
<Count>            := Count (ngram=<Count_ngram>)
<Count_ngram>      := discrete (min=1, max=3)
<TfIdf>            := TfIdf (ngram=<TfIdf_ngram>, use_idf=<TfIdf_use_idf>)
<TfIdf_ngram>      := discrete (min=1, max=3)
<TfIdf_use_idf>    := boolean ()
<Decomposer>       := <Noop> | <SVD>
<Noop>             := Noop ()
<SVD>              := SVD (n=<SVD_n>)
<SVD_n>            := discrete (min=50, max=200)
<Classifier>       := <LR> | <SVM> | <DT> | <NB>
<LR>               := LR (penalty=<LR_penalty>, C=<LR_C>)
<LR_penalty>       := categorical (options=['l1', 'l2'])
<LR_C>             := continuous (min=0.1, max=10)
<SVM>              := SVM (kernel=<SVM_kernel>, C=<SVM_C>)
<SVM_kernel>       := categorical (options=['rbf', 'linear', 'poly'])
<SVM_C>            := continuous (min=0.1, max=10)
<DT>               := DT (criterion=<DT_criterion>)
<DT_criterion>     := categorical (options=['gini', 'entropy'])
<NB>               := NB (var_smoothing=<NB_var_smoothing>)
<NB_var_smoothing> := continuous (min=1e-10, max=0.1)
```

Now we can start to see the power of the class-based API. Just with a few annotations
in the same classes that we anyway have to write, we automatically obtain a computational representation
(a grammar) that knows how to build infinitely many of these instances.
Futhermore, this works with any level of complexity, whether our classes receive
simple arguments (such as integers, floats, strings) or instances of other classes, and so on.

Let's take a look at how different pipelines can be generated with this grammar by sampling
10 random pipelines.

```python
for _ in range(10):
    print(grammar.sample())
```

You should see something like this, but your exact pipelines will be different due to random sampling.

```bash
Pipeline(classifier=SVM(C=4.09762837283166, kernel='rbf'), decomposer=Noop(),
         vectorizer=Count(ngram=1))
Pipeline(classifier=DT(criterion='entropy'), decomposer=Noop(),
         vectorizer=Count(ngram=2))
Pipeline(classifier=LR(C=5.309978916527087, penalty='l2'), decomposer=Noop(),
         vectorizer=Count(ngram=3))
Pipeline(classifier=LR(C=9.776994352626533, penalty='l1'), decomposer=Noop(),
         vectorizer=TfIdf(ngram=2, use_idf=True))
Pipeline(classifier=SVM(C=5.973033047496386, kernel='rbf'),
         decomposer=SVD(n=197), vectorizer=Count(ngram=3))
Pipeline(classifier=NB(var_smoothing=0.07941925220053651),
         decomposer=SVD(n=183), vectorizer=TfIdf(ngram=3, use_idf=False))
Pipeline(classifier=DT(criterion='entropy'), decomposer=SVD(n=144),
         vectorizer=TfIdf(ngram=1, use_idf=True))
Pipeline(classifier=SVM(C=6.052775609636756, kernel='poly'),
         decomposer=SVD(n=160), vectorizer=TfIdf(ngram=1, use_idf=True))
Pipeline(classifier=DT(criterion='entropy'), decomposer=Noop(),
         vectorizer=Count(ngram=1))
Pipeline(classifier=DT(criterion='entropy'), decomposer=Noop(),
         vectorizer=TfIdf(ngram=3, use_idf=True))
```

## Finding the best pipeline

To continue with the example, we will now search for the best pipeline.
We will evaluate our pipelines on the `movie_reviews` corpus. For that purpose
we need a fitness function, which is a simple callable that takes a pipeline and outputs
a score. Fortunately, the `movie_reviews.make_fn` function does this for us, taking
care of train/test splitting, fitting a pipeline in the training set and computing
the accuracy on the test set.

```python
fitness_fn = movie_reviews.make_fn(examples=100)
```

The `RandomSearch` strategy simply calls `grammar.sample()` a bunch of times
and stores the best performing pipeline. It has no intelligence whatsoever,
but it serves as a good baseline implementation.

We will run it for a total of `1000` fitness evaluations, or equivalently, a total
of `1000` different random pipelines.

```python
random_search = RandomSearch(grammar, fitness_fn, random_state=0)
best, score = random_search.run(1000)
```

!!! note
    For reproducibility purposes we can pass a fixed random seed in `random_state`.

## Final remarks

We only used `scikit-learn` here for illustrative purposes, but you can apply this strategy to any problem
whose solution consists of exploring a large space of complex class instances interrelated with each other.

Also, in this example we have manually written wrappers for `scikit-learn` classes to provide the necessary
annotations. However, specifically for `scikit-learn`, we already provide a bunch of wrappers with suitable
annotations in [`autogoal.contrib.sklearn`](/api/sklearn/).

We also only use `RandomSearch` in this example because the focus is on defining the pipelines.
However, the [`autogoal.search`](/api/search/) namespace contains other search strategies that perform
much better than plain random sampling.

