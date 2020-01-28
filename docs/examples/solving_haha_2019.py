from autogoal.ml import AutoClassifier
from autogoal.datasets import haha
from autogoal.search import Logger, PESearch, ConsoleLogger, ProgressLogger, MemoryLogger
from autogoal.kb import List, Sentence, Tuple, CategoricalVector
from autogoal.contrib import find_classes

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--iterations", type=int, default=1000)
parser.add_argument("--timeout", type=int, default=60)
parser.add_argument("--memory", type=int, default=1)
parser.add_argument("--popsize", type=int, default=10)

args = parser.parse_args()

print(args)

for cls in find_classes():
    print("Using: %s" % cls.__name__)

classifier = AutoClassifier(
    input=List(Sentence()),
    search_algorithm=PESearch,
    search_iterations=args.iterations,
    search_kwargs=dict(pop_size=args.popsize, evaluation_timeout=args.timeout, memory_limit=args.memory * 1024 ** 3),
)

class ErrorLogger(Logger):
    def error(self, e:Exception, solution):
        if e and solution:
            with open("errors.log", "a") as fp:
                fp.write(f"solution={repr(solution)}\nerror={repr(e)}\n\n")


memory_logger = MemoryLogger()

X_train, X_test, y_train, y_test = haha.load()

classifier.fit(X_train, y_train, logger=[ErrorLogger(), ConsoleLogger(), ProgressLogger(), memory_logger])
score = classifier.score(X_test, y_test)
print(score)

print(memory_logger.generation_best_fn)
print(memory_logger.generation_mean_fn)
