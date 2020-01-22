from autogoal.ml import AutoClassifier
from autogoal.datasets import cars
from autogoal.search import Logger, RandomSearch, PESearch, ConsoleLogger, ProgressLogger, MemoryLogger

classifier = AutoClassifier(
    search_algorithm=RandomSearch,
    search_iterations=1000,
    search_kwargs=dict(pop_size=10, evaluation_timeout=60, memory_limit=1024 ** 3),
)

X, y = cars.load()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
logger = MemoryLogger()

classifier.fit(X_train, y_train, logger=[ProgressLogger(), logger])
score = classifier.score(X_test, y_test)

print(score)
print(logger.generation_best_fn)
print(logger.generation_mean_fn)
