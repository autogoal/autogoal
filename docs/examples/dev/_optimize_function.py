from autogoal import optimize
from autogoal.grammar import Continuous, Discrete
from autogoal.utils import nice_repr
from autogoal.search import ConsoleLogger, ProgressLogger

import math


@nice_repr
class A:
    def __init__(self, x: Discrete(-1, 1)):
        self.x = x


def fn(a: A, y: Discrete(-1, 1)):
    return math.sin(a.x ** 2 - y ** 2)


best, best_fn = optimize(
    fn,
    logger=[ConsoleLogger(), ProgressLogger()],
    allow_duplicates=False,
)
print(best, best_fn)
