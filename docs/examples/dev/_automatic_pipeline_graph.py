from autogoal.kb import Tuple, build_pipeline_graph, build_pipelines, List, Vector, Matrix, Sentence, Word
from autogoal.utils import nice_repr

import logging

logging.basicConfig(level=logging.DEBUG)

@nice_repr
class A:
    def run(self, input: Sentence()) -> List(Word()):
        pass

@nice_repr
class B:
    def run(self, input: List(Word())) -> List(Vector()):
        pass


@nice_repr
class C:
    def run(self, input: List(Vector())) -> Matrix():
        pass


builder = build_pipelines(
    input=Tuple(Sentence(), Vector()),
    output=Matrix(),
    registry=[A, B, C]
)

pipeline = builder.sample()
print(pipeline)
print(pipeline.run([[[True], [False, True]]]))
