from autogoal import optimize
from autogoal.grammar import Continuous


def fn(x: Continuous(-1, 1), y: Continuous(-1, 1)):
    return -(x ** 2) - 2 * y ** 2 - 5 * x * y


best, best_fn = optimize(fn, iterations=100)
print(best, best_fn)

