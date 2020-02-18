from autogoal import optimize
from autogoal.grammar import Continuous
from autogoal.utils import nice_repr


@nice_repr
class A:
    def __init__(self, x: Continuous(0,1)):
        self.x = x


def fn(a: A, y: Continuous(-1, 1)):
    return a.x + y


best, best_fn = optimize(fn, iterations=100)
print(best, best_fn)
