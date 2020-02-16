# # ICML 2020 example in the HAHA challenge

# This script runs an instance of [`AutoClassifier`](/api/autogoal.ml#AutoClassifier)
# in the HAHA 2019 challenge.
# The results obtained were published in the paper presented at ICML 2020.

# Most of this example follows the same logic as the [ICML UCI example](/examples/solving_uci_datasets).
# First the necessary imports

from autogoal.ml import AutoML
from autogoal.datasets import meddocan
from autogoal.search import (
    Logger,
    PESearch,
    ConsoleLogger,
    ProgressLogger,
    MemoryLogger,
)
from autogoal.kb import List, Sentence, Word, Postag

# Next, we parse the command line arguments to configure the experiment.

# The default values are the ones used for the experimentation reported in the paper.

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--iterations", type=int, default=10000)
parser.add_argument("--timeout", type=int, default=1800)
parser.add_argument("--memory", type=int, default=20)
parser.add_argument("--popsize", type=int, default=50)
parser.add_argument("--selection", type=int, default=10)
parser.add_argument("--global-timeout", type=int, default=None)
parser.add_argument("--examples", type=int, default=None)
parser.add_argument("--token", default=None)
parser.add_argument("--channel", default=None)

args = parser.parse_args()

print(args)

# Instantiate the classifier.
# Note that the input and output types here are defined to match the problem statement,
# i.e., entity recognition.

classifier = AutoML(
    search_algorithm=PESearch,
    input=List(List(Word())),
    output=List(List(Postag())),
    search_iterations=args.iterations,
    score_metric=meddocan.F1_beta,
    cross_validation_steps=1,
    exclude_filter=".*Word2Vec.*",
    search_kwargs=dict(
        pop_size=args.popsize,
        search_timeout=args.global_timeout,
        evaluation_timeout=args.timeout,
        memory_limit=args.memory * 1024 ** 3,
    ),
)

# This custom logger is used for debugging purposes, to be able later to recover
# the best pipelines and all the errors encountered in the experimentation process.

class CustomLogger(Logger):
    def error(self, e: Exception, solution):
        if e and solution:
            with open("meddocan_errors.log", "a") as fp:
                fp.write(f"solution={repr(solution)}\nerror={e}\n\n")

    def update_best(self, new_best, new_fn, *args):
        with open("meddocan.log", "a") as fp:
            fp.write(f"solution={repr(new_best)}\nfitness={new_fn}\n\n")

# Basic logging configuration.

logger = MemoryLogger()
loggers = [ProgressLogger(), ConsoleLogger(), logger]

if args.token:
    from autogoal.contrib.telegram import TelegramLogger

    telegram = TelegramLogger(
        token=args.token,
        name=f"MEDDOCAN",
        channel=args.channel,
    )
    loggers.append(telegram)

# Finally, loading the MEDDOCAN dataset, running the `AutoClassifier` instance,
# and printing the results.

X_train, X_test, y_train, y_test = meddocan.load(max_examples=args.examples)

classifier.fit(X_train, y_train, logger=loggers)
score = classifier.score(X_test, y_test)

print(score)
print(logger.generation_best_fn)
print(logger.generation_mean_fn)
