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

classifier = AutoML(
    search_algorithm=PESearch,
    input=List(List(Word())),
    output=List(List(Postag())),
    search_iterations=args.iterations,
    score_metric=meddocan.precision,
    cross_validation_steps=1,
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
            with open("meddocan_errors.log", "a") as fp:
                fp.write(f"solution={repr(solution)}\nerror={e}\n\n")

    def update_best(self, new_best, new_fn, *args):
        with open("meddocan.log", "a") as fp:
            fp.write(f"solution={repr(new_best)}\nfitness={new_fn}\n\n")


memory_logger = MemoryLogger()

X_train, X_test, y_train, y_test = meddocan.load(max_examples=args.examples)

classifier.fit(
    X_train,
    y_train,
    logger=[CustomLogger(), ConsoleLogger(), ProgressLogger(), memory_logger],
)
score = classifier.score(X_test, y_test)

print(score)
print(memory_logger.generation_best_fn)
print(memory_logger.generation_mean_fn)
