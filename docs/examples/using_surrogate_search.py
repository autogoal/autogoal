from autogoal.search import SurrogateSearch, PESearch, ConsoleLogger
from autogoal.grammar import Continuous, Sampler, generate_cfg
from autogoal.utils import nice_repr


@nice_repr
class Input:
    def __init__(
        self, x: Continuous(-1, 1), y: Continuous(-1, 1), z: Continuous(-1, 1)
    ):
        self.x = x
        self.y = y
        self.z = z


def fn(t: Input):
    return -(t.x ** 2) - (t.y ** 2) - (t.z ** 2)


search = SurrogateSearch(PESearch, None, 10, generator_fn=generate_cfg(Input), fitness_fn=fn)

search.run(1000, ConsoleLogger())
