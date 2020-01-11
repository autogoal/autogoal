from autogoal.grammar import Sampler
from autogoal.search import RandomSearch, PESearch


def generate(sampler: Sampler):
    x1 = sampler.continuous(0, 1, "x1")
    x2 = sampler.continuous(0, 1, "x2")

    if x1 > x2:
        return (x1, x2)

    return (0, 0)


def fn(t):
    x1, x2 = t
    return x1 * x2


search = PESearch(generate, fn)
best, y = search.run(1000)

print(search._model)

print(best, y)
