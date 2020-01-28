# Solving UCI datasets

This script runs an instance of [`AutoClassifier`](/api/autogoal.ml#AutoClassifier)
in anyone of the UCI datasets available.

We will need `argparse` for passing arguments to the script.

```python
import argparse
```

From `sklearn` we will use `train_test_split` to build train and validation sets.

```python
from sklearn.model_selection import train_test_split
```

From `autogoal` we need a bunch of datasets.

```python
from autogoal import datasets
from autogoal.datasets import abalone, cars, dorothea, gisette, shuttle, yeast
```

We will also import this annotation type.

```python
from autogoal.kb import MatrixContinuousDense
```

This is the real deal, the class `AutoClassifier` does all the work.

```python
from autogoal.ml import AutoClassifier
```

And from the `autogoal.search` module we will need a couple logging utilities
and the `PESearch` class.

```python
from autogoal.search import (
    ConsoleLogger,
    Logger,
    MemoryLogger,
    PESearch,
    ProgressLogger,
)
```

## Parsing arguments

Here we simply receive a bunch of arguments from the command line
to decide which dataset to run and hyperparameters.
They should be pretty self-explanatory.

```python
parser = argparse.ArgumentParser()
parser.add_argument("--iterations", type=int, default=1000)
parser.add_argument("--timeout", type=int, default=60)
parser.add_argument("--memory", type=int, default=1)
parser.add_argument("--popsize", type=int, default=10)
```

The most important argument is this one, which selects the dataset.

```python
parser.add_argument(
    "-d",
    "--dataset",
    choices=["abalone", "cars", "dorothea", "gisette", "shuttle", "yeast"],
)

args = parser.parse_args()
```

## Loading the data

Here we dynamically load the corresponding dataset and,
if necesary, split it into training and testing sets.

```python
data = getattr(datasets, args.dataset).load()

if len(data) == 4:
    X_train, X_test, y_train, y_test = data
else:
    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
```

## Running the AutoML

Finally we can instantiate out `AutoClassifier` with all the custom
paramenters we received from the command line.

```python
classifier = AutoClassifier(
    input=MatrixContinuousDense(),
    search_algorithm=PESearch,
    search_iterations=args.iterations,
    search_kwargs=dict(
        pop_size=10,
        evaluation_timeout=args.timeout,
        memory_limit=args.memory * 1024 ** 3,
    ),
)
```

And run it.

```python
logger = MemoryLogger()
classifier.fit(X_train, y_train, logger=[ProgressLogger(), ConsoleLogger(), logger])
score = classifier.score(X_test, y_test)
```

Let's see how it went.

```python
print(score)
print(logger.generation_best_fn)
print(logger.generation_mean_fn)
```

