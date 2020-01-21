from autogoal.ml import AutoTextClassifier
from autogoal.datasets import haha
from autogoal.search import ConsoleLogger, PESearch

classifier = AutoTextClassifier(
    search_algorithm=PESearch,
    search_iterations=1000,
    search_kwargs=dict(pop_size=10, evaluation_timeout=60, memory_limit=1024 ** 3),
)

X_train, X_test, y_train, y_test = haha.load()

classifier.fit(X_train, y_train, logger=ConsoleLogger())
score = classifier.score(X_test, y_test)
print(score)
