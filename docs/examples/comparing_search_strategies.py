# # Comparing Search Strategies
#
# This example compares the performance of [RandomSearch](/api/autogoal.search/#RandomSearch)
# and [PESearch](/api/autogoal.search/#PESearch) on a toy problem.

from autogoal.search import RandomSearch, PESearch, ConsoleLogger

# The problem to solve consists of a simple puzzle:
# Combining the digits 1 through 9 in different operations
# to obtain the largest possible value.
# We can apply addition, substraction, multiplication, division and exponentiation.

# To model all the possible operations and operators we will design a simple grammar
# using the [class API](/guide/cfg.md)

from autogoal.grammar import generate_cfg, Union
from autogoal.utils import nice_repr


def evaluate(expr):
    def stream():
        for i in range(1, 10):
            yield i

        while True:
            yield 0

    return expr(stream())


@nice_repr
class Number:
    def __call__(self, stream):
        return next(stream)


class Operator:
    def __init__(self, left: "Expr", right: "Expr"):
        self.left = left
        self.right = right

    def __call__(self, stream):
        return self.operate(self.left(stream), self.right(stream))


@nice_repr
class Add(Operator):
    def operate(self, left, right):
        return left + right


@nice_repr
class Mult(Operator):
    def operate(self, left, right):
        return left * right


@nice_repr
class Concat(Operator):
    def operate(self, left, right):
        return int(str(left) + str(right))


Expr = Union("Expr", Number, Add, Mult, Concat)

grammar = generate_cfg(Expr)

print(grammar)

for i in range(100):
    try:
        solution = grammar.sample()
        print(solution)
    except ValueError:
        continue


search_rand = RandomSearch(grammar, evaluate, errors='ignore')
best_rand, best_fn_rand = search_rand.run(1000, logger=ConsoleLogger())

search_pe = PESearch(grammar, evaluate, pop_size=10, errors='ignore')
best_pe, best_fn_pe = search_pe.run(1000, logger=ConsoleLogger())

print(best_rand, best_fn_rand)
print(best_pe, best_fn_pe)

print(search_pe._model)
