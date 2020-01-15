# from autogoal.kb import (
#     Word,
#     List,
#     CategoricalVector,
# )
# from autogoal.kb import build_pipelines
# from autogoal.datasets import movie_reviews
# from autogoal.contrib.sklearn import find_classes

# pipeline_generator = build_pipelines(
#     input=List(Word()),
#     output=CategoricalVector(),
#     registry=find_classes(r"(.*Classifier|.*Vectorizer)"),
# )

# fitness_fn = movie_reviews.make_fn(examples=100)

# from autogoal.search import RandomSearch

# search = RandomSearch(pipeline_generator, fitness_fn, errors='warn')
# best, best_score = search.run(100)

# print(best, best_score)

from autogoal.kb import Tuple, build_pipelines
from autogoal.utils import nice_repr

@nice_repr
class A:
    def run(self, input: Tuple(str, float)) -> int:
        return len(input[0]) * input[1]

@nice_repr
class B:
    def run(self, input: bool) -> str:
        return str(input)

builder = build_pipelines(
    input=Tuple(bool, float),
    output=int,
    registry=[A, B]
)

pipeline = builder.sample()
print(pipeline.run((True, 3.5)))
