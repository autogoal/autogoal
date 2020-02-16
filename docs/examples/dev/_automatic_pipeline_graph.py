from autogoal.kb import Tuple, build_pipelines, List
from autogoal.utils import nice_repr

@nice_repr
class A:
    def run(self, input: List(str)) -> int:
        return len(input)

@nice_repr
class B:
    def run(self, input: bool) -> str:
        return str(input)


builder = build_pipelines(
    input=List(List(List(bool))),
    output=List(List(int)),
    registry=[A, B]
)

pipeline = builder.sample()
print(pipeline)
print(pipeline.run([[[True], [False, True]]]))

