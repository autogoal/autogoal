# # Solving UCI datasets

# This script runs an instance of [`AutoML`](/api/autogoal.ml#automl)
# in anyone of the UCI datasets available.

# The datasets used in this experimentation are taken from the [UCI repository](https://archive.ics.uci.edu/ml/index.php).
# Concretely, the following datasets are used:

# | Dataset | URL |
# |--|--|
# | Abalone | <https://archive.ics.uci.edu/ml/datasets/Abalone> |
# | Cars | <https://archive.ics.uci.edu/ml/datasets/Car+Evaluation> |
# | Dorothea | <https://archive.ics.uci.edu/ml/datasets/dorothea> |
# | Gisette | <https://archive.ics.uci.edu/ml/datasets/Gisette> |
# | Shuttle | <https://archive.ics.uci.edu/ml/datasets/Statlog+(Shuttle)> |
# | Yeast | <https://archive.ics.uci.edu/ml/datasets/Yeast> |
# | German Credit | <https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)> |

# ## Experimentation parameters
#
# This experiment was run with the following parameters:
#
# | Parameter | Value |
# |--|--|
# | Total epochs         | 20     |
# | Maximum iterations   | 10000  |
# | Timeout per pipeline | 5 min  |
# | Global timeout       | 1 hour |
# | Max RAM per pipeline | 10 GB  |
# | Population size      | 100    |
# | Selection (k-best)   | 20     |
# | Early stop           | 200 iterations |

# The experiments were run in the following hardware configurations
# (allocated indistinctively according to available resources):

# | Config | CPU | Cache | Memory | HDD |
# |--|--|--|--|--|
# | **A** | 12 core Intel Xeon Gold 6126 | 19712 KB |  191927.2MB | 999.7GB  |
# | **B** | 6 core Intel Xeon E5-1650 v3 | 15360 KB |  32045.5MB  | 2500.5GB |
# | **C** | Quad core Intel Core i7-2600 |  8192 KB |  15917.1MB  | 1480.3GB |

# !!! note
#     The hardware configuration details were extracted with `inxi -CmD` and summarized.

# ## Relevant imports

# We will need `argparse` for passing arguments to the script and `json` for serialization of results.

import argparse
import json

# From `sklearn` we will use `train_test_split` to build train and validation sets.

from sklearn.model_selection import train_test_split

# From `autogoal` we need a bunch of datasets.

from autogoal import datasets
from autogoal.datasets import (
    abalone,
    cars,
    dorothea,
    gisette,
    shuttle,
    yeast,
    german_credit,
)

# We will also import this annotation type.

from autogoal.kb import MatrixContinuousDense, CategoricalVector

# This is the real deal, the class `AutoML` does all the work.

from autogoal.ml import AutoML

# And from the `autogoal.search` module we will need a couple logging utilities
# and the `PESearch` class.

from autogoal.search import (
    ConsoleLogger,
    Logger,
    MemoryLogger,
    PESearch,
    ProgressLogger,
)

# ## Parsing arguments

# Here we simply receive a bunch of arguments from the command line
# to decide which dataset to run and hyperparameters.
# They should be pretty self-explanatory.

# The default values are the ones used for the experimentation reported in the paper.

parser = argparse.ArgumentParser()
parser.add_argument("--iterations", type=int, default=10000)
parser.add_argument("--timeout", type=int, default=300)
parser.add_argument("--memory", type=int, default=10)
parser.add_argument("--popsize", type=int, default=100)
parser.add_argument("--selection", type=int, default=20)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--global-timeout", type=int, default=60 * 60)
parser.add_argument("--early-stop", type=int, default=200)
parser.add_argument("--token", default=None)
parser.add_argument("--channel", default=None)
parser.add_argument("--target", default=1.0, type=float)

# The most important argument is this one, which selects the dataset.

valid_datasets = [
    "abalone",
    "cars",
    "dorothea",
    "gisette",
    "shuttle",
    "yeast",
    "german_credit",
]

parser.add_argument("--dataset", choices=valid_datasets + ["all"], default="all")

args = parser.parse_args()

# ## Experimentation

# Now we run all the experiments, in each of the datasets selected,
# for as many epochs as specified in the command line.

if args.dataset != "all":
    valid_datasets = [args.dataset]

for epoch in range(args.epochs):
    for dataset in valid_datasets:
        print("=============================================")
        print(" Running dataset: %s - Epoch: %s" % (dataset, epoch))
        print("=============================================")
        data = getattr(datasets, dataset).load()

# Here we dynamically load the corresponding dataset and,
# if necesary, split it into training and testing sets.

        if len(data) == 4:
            X_train, X_test, y_train, y_test = data
        else:
            X, y = data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Finally we can instantiate out `AutoML` with all the custom
# parameters we received from the command line.

        classifier = AutoML(
            output=CategoricalVector(),
            search_algorithm=PESearch,
            search_iterations=args.iterations,
            search_kwargs=dict(
                pop_size=args.popsize,
                selection=args.selection,
                evaluation_timeout=args.timeout,
                memory_limit=args.memory * 1024 ** 3,
                early_stop=args.early_stop,
                search_timeout=args.global_timeout,
                target_fn=args.target,
            ),
        )

# Here we configure all the logging strategies we will use.
# `MemoryLogger` stores each generation's info in a list that we can
# later dump into a JSON log file.
# `ProgressLogger` and `ConsoleLogger` are for pretty printing the results on the console.

        logger = MemoryLogger()
        loggers = [ProgressLogger(), ConsoleLogger(), logger]

# `TelegramLogger` outputs debug information to a custom Telegram channel, if configured.

        if args.token:
            from autogoal.contrib.telegram import TelegramLogger

            telegram = TelegramLogger(
                token=args.token,
                name=f"UCI dataset=`{dataset}` run=`{epoch}`",
                channel=args.channel,
            )
            loggers.append(telegram)

# Finally, we run the AutoML classifier once and compute the score on an independent test-set.

        classifier.fit(X_train, y_train, logger=loggers)
        score = classifier.score(X_test, y_test)

        print(score)
        print(logger.generation_best_fn)
        print(logger.generation_mean_fn)

# And store the results on a log file.

        with open("uci_datasets.log", "a") as fp:
            fp.write(
                json.dumps(
                    dict(
                        dataset=dataset,
                        score=score,
                        generation_best=logger.generation_best_fn,
                        generation_mean=logger.generation_mean_fn,
                        best_pipeline=repr(classifier.best_pipeline_),
                        search_iterations=args.iterations,
                        search_kwargs=dict(
                            pop_size=args.popsize,
                            selection=args.selection,
                            evaluation_timeout=args.timeout,
                            memory_limit=args.memory * 1024 ** 3,
                            early_stop=args.early_stop,
                            search_timeout=args.global_timeout,
                        ),
                    )
                )
            )
            fp.write("\n")
