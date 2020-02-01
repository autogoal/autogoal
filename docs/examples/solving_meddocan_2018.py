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


classifier = AutoML(
    input=List(List(Word())),
    output=List(List(Postag())),
    search_iterations=10000,
    search_kwargs=dict(pop_size=10, evaluation_timeout=60, memory_limit=1024 ** 3),
    score_metric=meddocan.precision,
    # errors="raise",
    exclude_filter=".*Keras.*",
)


class CustomLogger(Logger):
    def error(self, e: Exception, solution):
        if e and solution:
            with open("meddocan_errors.log", "a") as fp:
                fp.write(f"solution={repr(solution)}\nerror={repr(e)}\n\n")

    def update_best(self, new_best, new_fn, *args):
        with open("meddocan.log", "a") as fp:
            fp.write(f"solution={repr(new_best)}\nfitness={new_fn}\n\n")


memory_logger = MemoryLogger()

X_train, X_test, y_train, y_test = meddocan.load_corpus()

classifier.fit(
    X_train, y_train, logger=[CustomLogger(), ConsoleLogger(), memory_logger]
)
score = classifier.score(X_test, y_test)

print(score)
print(memory_logger.generation_best_fn)
print(memory_logger.generation_mean_fn)
