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

# load fitness
fn = movie_reviews.make_fn() 

# automl and io types
automl = AutoML(
    input(Seq[Sentence], Supervised[Categorical]),
    output=Seq[Sentence],
    registry=[SimulateTypos] + find_classes(), # using a augment model on the training data
    evaluation_timeout=Min,
    memory_limit=4 * Gb,
    search_timeout=Min,
)

acc = fn(automl)
print(acc)