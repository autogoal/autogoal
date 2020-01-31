from autogoal.ml import AutoClassifier
from autogoal.datasets import haha
from autogoal.search import (
    Logger,
    PESearch,
    ConsoleLogger,
    ProgressLogger,
    MemoryLogger,
)
from autogoal.kb import List, Sentence, Tuple, CategoricalVector
from autogoal.contrib import find_classes
from sklearn.metrics import f1_score

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

args = parser.parse_args()

print(args)

for cls in find_classes():
    print("Using: %s" % cls.__name__)


classifier = AutoClassifier(
    search_algorithm=PESearch,
    input=List(Sentence()),
    search_iterations=args.iterations,
    score_metric=f1_score,
    search_kwargs=dict(
        pop_size=args.popsize,
        search_timeout=args.global_timeout,
        evaluation_timeout=args.timeout,
        memory_limit=args.memory * 1024 ** 3,
    ),
)


class CustomLogger(Logger):
    def error(self, e: Exception, solution):
        if e and solution:
            with open("haha_errors.log", "a") as fp:
                fp.write(f"solution={repr(solution)}\nerror={repr(e)}\n\n")

    def update_best(self, new_best, new_fn, *args):
        with open("haha.log", "a") as fp:
            fp.write(f"solution={repr(new_best)}\nfitness={new_fn}\n\n")


memory_logger = MemoryLogger()
loggers = [ProgressLogger(), ConsoleLogger(), CustomLogger(), memory_logger]

if args.token:
    from autogoal.contrib.telegram import TelegramBotLogger

    telegram = TelegramBotLogger(token=args.token, name=f"HAHA 2019")
    loggers.append(telegram)

X_train, X_test, y_train, y_test = haha.load(max_examples=args.examples)

classifier.fit(X_train, y_train, logger=loggers)
score = classifier.score(X_test, y_test)


print(score)
print(memory_logger.generation_best_fn)
print(memory_logger.generation_mean_fn)
