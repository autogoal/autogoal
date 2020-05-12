# Comparing Search Strategies

This example compares the performance of [RandomSearch](/api/autogoal.search/#RandomSearch)
and [PESearch](/api/autogoal.search/#PESearch) on a toy problem.

```python
from autogoal.search import RandomSearch, PESearch, ConsoleLogger, ProgressLogger
```

The problem to solve consists of a simple puzzle:
Combining the digits 1 through 9 in different operations
to obtain the largest possible value.
We can apply addition, substraction, multiplication, division and exponentiation.

To model all the possible operations and operators we will design a simple grammar
using the [class API](/guide/cfg.md)

```python
from autogoal.grammar import generate_cfg, Union
from autogoal.utils import nice_repr


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
```

Our grammar is composed of addition, multiplication and concatenation operators.
Here are some possible examples:

```python
for i in range(100):
    try:
        solution = grammar.sample()
        print(solution)
    except ValueError:
        continue
```

To evaluate how good a formula is, we simply feed the expression instance
with a sequence of numbers from 1 to 9. If the expression requires more
than 9 digits, it results in an error. The actual value of performing
corresponding operations is done in the `__call__` method of the expression classes.

```python
def evaluate(expr):
    def stream():
        for i in range(1, 10):
            yield i

        raise ValueError("Too many values asked")

    return expr(stream())
```

We will run 1000 iterations of each search strategy to compare their long-term performance.

```python
search_rand = RandomSearch(grammar, evaluate, errors='ignore')
best_rand, best_fn_rand = search_rand.run(1000, logger=[ConsoleLogger(), ProgressLogger()])

search_pe = PESearch(grammar, evaluate, pop_size=10, errors='ignore')
best_pe, best_fn_pe = search_pe.run(1000, logger=[ConsoleLogger(), ProgressLogger()])
```

And here are the results.

```python
print(best_rand, best_fn_rand)
print(best_pe, best_fn_pe)
```

