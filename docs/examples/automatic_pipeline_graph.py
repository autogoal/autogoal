from autogoal.kb import (
    Word,
    List,
    CategoricalVector,
)
from autogoal.kb import build_pipelines
from autogoal.datasets import movie_reviews
from autogoal.contrib.sklearn import find_classes

pipeline_generator = build_pipelines(
    input=List(Word()),
    output=CategoricalVector(),
    registry=find_classes(".*Classifier") + find_classes(".*Vectorizer"),
)

fitness_fn = movie_reviews.make_fn(examples=100)

from autogoal.search import RandomSearch

search = RandomSearch(pipeline_generator, fitness_fn, errors='warn')
best, best_score = search.run(100)

print(best, best_score)
