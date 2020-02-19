# # Functional API

# AutoGOAL's functional API allows you to transform any Python callable (e.g., a method)
# into an optimizable target. In contrast with the [class-based](/guide/cfg/) and [graph-based](/guide/graph/) APIs,
# the functional API does not require to know before-hand the structure of the space you want to optimize.

# This enables very flexible use cases, in which you can iterate quickly, experiment, and transform deterministic code to solve
# one particular task into optimizable software that seems to magically solve the problem for you in the best possible way.

# Let's start with a toy example just to show the basic usage of the API.

from autogoal.sampling import Sampler


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
