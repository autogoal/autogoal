from autogoal.ml import AutoClassifier
from autogoal.datasets import meddocan
from autogoal.search import Logger, PESearch, ConsoleLogger, ProgressLogger, MemoryLogger
from autogoal.kb import List, Sentence, Word, Postag
from autogoal.contrib.keras import KerasSequenceTagger
from autogoal.contrib.gensim._base import Word2VecEmbeddingSpanish
from autogoal.contrib._wrappers import MatrixBuilder, TensorBuilder

classifier = AutoClassifier(
    input=List(List(Word())), #input: Untagged tokenized documents
    output=List(List(Postag())),
    search_algorithm=PESearch,
    search_iterations=1000,
    search_kwargs=dict(pop_size=10, evaluation_timeout=60, memory_limit=1024 ** 3),
    score_metric=meddocan.precision,
    registry=[Word2VecEmbeddingSpanish, KerasSequenceTagger, MatrixBuilder, TensorBuilder]
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

classifier.fit(X_train, y_train, logger=[CustomLogger(), ConsoleLogger(), memory_logger])
score = classifier.score(X_test, y_test)
print(score)

print(memory_logger.generation_best_fn)
print(memory_logger.generation_mean_fn)
