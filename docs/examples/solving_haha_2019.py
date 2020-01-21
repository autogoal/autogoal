from autogoal.ml import AutoTextClassifier
from autogoal.datasets import haha
from autogoal.search import ConsoleLogger, PESearch

classifier = AutoTextClassifier(
    search_algorithm=PESearch,
    search_iterations=1000,
    search_kwargs=dict(pop_size=10, evaluation_timeout=60, memory_limit=1024 ** 3),
)

class CustomLogger(ConsoleLogger):
    def error(self, e:Exception, solution):
        super().error(e, solution)

        if e and solution:
            with open("errors.log", "a") as fp:
                fp.write(f"solution={repr(solution)}\nerror={repr(e)}\n\n")


X_train, X_test, y_train, y_test = haha.load()

classifier.fit(X_train, y_train, logger=CustomLogger())
score = classifier.score(X_test, y_test)
print(score)
