# Mandatory and optional dependencies

!!! note
    This section is not relevant if you downloaded the Docker image of AutoGOAL.

AutoGOAL itself does not depends on any third-party machine learning or optimization framework. The only real **mandatory** dependencies are:

    black = "^19.10b0"
    enlighten = "^1.4.0"
    networkx = "^2.4"
    numpy = "^1.19.2"
    psutil = "^5.6.7"
    pydot = "^1.4.1"
    pyyaml = "^5.2"
    scipy = "^1.5.2"
    termcolor = "^1.1.0"
    toml = "^0.10.0"
    tqdm = "^4.50.2"
    typer = "^0.3.2"

If you simply install with:

    pip install autogoal

Then you will be able to run almost none of the examples, since most of them use external dependencies such as `keras` or `sklearn`. If you want to install all dependencies, use:

    pip install autogoal[contrib]

Currently, first level optional dependencies include:

    gensim = "^3.8.1"
    jupyterlab = "^1.2.4"
    keras = "^2.3.1"
    nltk = "^3.4.5"
    nx_altair = "^0.1.4"
    python-telegram-bot = "^12.4.2"
    scikit-learn = "^0.22"
    seqlearn = "^0.2"
    sklearn_crfsuite = "^0.3.6"
    spacy = "^2.2.3"
    streamlit = "^0.59.0"
    transformers = "^2.3.0"
    wikipedia = "^1.4.0"

You can also hand-pick which of these dependencies to install. It depends on the use you want to make of AutoGOAL.

## What about development dependencies?

If you want to develop for the project then you will need the development dependencies. The recommended way to do this is to [use Docker and use our development image](../contributing).

## Why not include all the dependencies?

AutoGOAL itself does not depend on `sklearn` or `keras`, for example, and it's not necessary to have either of these frameworks to make use of AutoGOAL. You might have a completely different problem setup, using totally new frameworks, or even your own classes. Hence, it makes no sense to force users of AutoGOAL to carry with all these heavy dependencies.

Likewise, you may want a different version of `keras`, or even one that integrates with `pytorch` instead of `tensorflow`. AutoGOAL is agnostic to the underlying classes you use to actually build pipelines.

Finally, you can use AutoGOAL to optimize almost everything, not just machine learning pipelines. You can, for example, optimize the configuration parameters for a `scrapy` crawler, or automatically find the regular expression that best matches some patterns. These are just completely random examples to illustrate that you can use AutoGOAL in scenarios other than AutoML. Whenever you can describe the set of possible solutions to a problem as a grammar, AutoGOAL comes to your help.

## Then why are there optional dependencies?

Since AutoGOAL requires annotations and a specific use of its API to perform its magic, we provide pre-defined wrappers for `sklearn`, `keras` and `nltk`, to ease the development in the main use case of AutoGOAL, which is AutoML. We, the developers, are ourselves researchers in the AutoML area, and as such, we use AutoGOAL for this purpose. Anytime we need a new framework in our experiments, we add the corresponding wrappers to AutoGOAL to help the next generation of machine learning researchers and practicioners.

## Will missing optional dependencies bite me?

They should not. All the code inside `autogoal`, except for `autogoal.contrib` is independent of any machine learning framework. The `autogoal.contrib` namespace contains all the code that depends on third-party libraries, and it's composed mostly of suitably annotated wrappers for these frameworks. For example, in `autogoal.contrib.keras` you will find utilities for automatically creating keras-based neural networks. When you import any of the modules in `autogoal.contrib.*`, the first thing we do is attempt to import the corresponding dependencies and provide helpful error messages otherwise.

For example, if you don't have `keras` installed and attempt to use:

```python
from autogoal.contrib.keras import KerasClassifier
```

You will get the following error:

```bash
(!) Code in `autogoal.contrib.keras` requires `keras` installed.
(!) Run `pip install -U autogoal[keras]` to get it.
```

Running `pip install autogoal[keras]` installs all keras-dependent third-party dependencies. Likewise, you can `pip install autogoal[module]` for every `module` in `autogoal.contrib.*`.
