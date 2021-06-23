from autogoal.logging import logger
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

# load data
X, y = movie_reviews.load() 

# automl and io types
automl = AutoML(
    input=(Seq[Sentence], Supervised[Categorical]),
    output=Seq[Sentence],
    registry=[SimulateTypos] + find_classes(),
    evaluation_timeout=Min,
    memory_limit=2.5 * Gb,
    search_timeout=Min,
)

# transform the training set X
automl.fit(X, y, logger=[RichLogger()])
augX = automl.predict(X)
print(augX[:3])