# # ICML 2020 UCI datasets

# This script runs an instance of [`AutoClassifier`](/api/autogoal.ml#AutoClassifier)
# in anyone of the UCI datasets available.
# The results obtained were published in the paper presented at ICML 2020.

# We will need `argparse` for passing arguments to the script and `json` for serialization of results.

import argparse
import json

# From `sklearn` we will use `train_test_split` to build train and validation sets.

from sklearn.model_selection import train_test_split

# From `autogoal` we need a bunch of datasets.

from autogoal import datasets
from autogoal.datasets import abalone, cars, dorothea, gisette, shuttle, yeast

# We will also import this annotation type.

from autogoal.kb import MatrixContinuousDense

# This is the real deal, the class `AutoClassifier` does all the work.

from autogoal.ml import AutoClassifier

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

parser = argparse.ArgumentParser()
parser.add_argument("--iterations", type=int, default=1000)
parser.add_argument("--timeout", type=int, default=60)
parser.add_argument("--memory", type=int, default=1)
parser.add_argument("--popsize", type=int, default=10)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--global-timeout", type=int, default=60 * 60)
parser.add_argument("--early-stop", type=int, default=100)
parser.add_argument("--token", default=None)

# The most important argument is this one, which selects the dataset.

valid_datasets = ["abalone", "cars", "dorothea", "gisette", "shuttle", "yeast"]

parser.add_argument("--dataset", choices=valid_datasets + ["all"], default="all")

args = parser.parse_args()

# ## Loading the data

# Here we dynamically load the corresponding dataset and,
# if necesary, split it into training and testing sets.

if args.dataset != "all":
    valid_datasets = [args.dataset]

for epoch in range(args.epochs):
    for dataset in valid_datasets:
        print("=============================================")
        print(" Running dataset: %s - Epoch: %s" % (dataset, epoch))
        print("=============================================")
        data = getattr(datasets, dataset).load()

        if len(data) == 4:
            X_train, X_test, y_train, y_test = data
        else:
            X, y = data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        # ## Running the AutoML

        # Finally we can instantiate out `AutoClassifier` with all the custom
        # paramenters we received from the command line.

        classifier = AutoClassifier(
            input=MatrixContinuousDense(),
            search_algorithm=PESearch,
            search_iterations=args.iterations,
            search_kwargs=dict(
                pop_size=10,
                evaluation_timeout=args.timeout,
                memory_limit=args.memory * 1024 ** 3,
                early_stop=args.early_stop,
                search_timeout=args.global_timeout,
            ),
        )

        # And run it.

        logger = MemoryLogger()
        loggers = [ProgressLogger(), ConsoleLogger(), logger]

        if args.token:
            from autogoal.contrib.telegram import TelegramBotLogger

            telegram = TelegramBotLogger(token=args.token, name=f"ICML UCI dataset=`{dataset}` run=`{epoch}`")
            loggers.append(telegram)

        classifier.fit(X_train, y_train, logger=loggers)
        score = classifier.score(X_test, y_test)

        # Let's see how it went.

        print(score)
        print(logger.generation_best_fn)
        print(logger.generation_mean_fn)

        with open("uci_datasets.log", "a") as fp:
            fp.write(
                json.dumps(
                    dict(
                        dataset=dataset,
                        epoch=epoch,
                        score=score,
                        generation_best=logger.generation_best_fn,
                        generation_mean=logger.generation_mean_fn,
                        best_pipeline=repr(classifier.best_pipeline_),
                        search_iterations=args.iterations,
                        search_kwargs=dict(
                            pop_size=10,
                            evaluation_timeout=args.timeout,
                            memory_limit=args.memory * 1024 ** 3,
                            early_stop=args.early_stop,
                            search_timeout=args.global_timeout,
                        ),
                    )
                )
            )
            fp.write("\n")
