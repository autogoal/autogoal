from os import SCHED_OTHER
from autogoal.logging import logger
from autogoal import search
from autogoal.kb import (
    Sentence,
    Seq,
    Supervised,
    Categorical,
)

from autogoal.search import RichLogger
from autogoal.utils import Min, Gb
from autogoal.contrib import find_classes

from autogoal.ml import AutoML
from autogoal.experimental.augly import (
    SimulateTypos, # simulates typographical errors made by typing incorrect letters due to closeness
)

from autogoal.datasets import movie_reviews

# load dataset
sentences, classes = movie_reviews.load()

X_train, y_train, X_test, y_test = movie_reviews.make_fn()

# automl and io types
automl = AutoML(
    input(Seq[Sentence], Supervised[Categorical]),
    output=Seq[Sentence],
    registry=[SimulateTypos] + find_classes(), # using a augment model on the training data
    evaluation_timeout=Min,
    memory_limit =4 * Gb,
    search_timeout =Min,
)

automl.fit(X_train, y_train, logger=[RichLogger()])

score = automl.score(X_test, y_test)
print(f"Score: {score}")