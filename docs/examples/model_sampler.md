```python
from autogoal.search import ModelSampler
from autogoal.grammar import (
    generate_cfg,
    Union,
    Discrete,
    Continuous,
    Boolean,
    Categorical,
)


class A:
    def __init__(self, a: Discrete(1, 5), b: Boolean()):
        self.a = a
        self.b = b

    def __repr__(self):
        return "A(a=%r, b=%r)" % (self.a, self.b)


class B:
    def __init__(self, c:Continuous(0,1), d:Categorical('X', 'Y')):
        self.c = c
        self.d = d

    def __repr__(self):
        return "B(c=%r, d=%r)" % (self.c, self.d)


class C:
    def __init__(self, x: Union("AB", A, B)):
        self.x = x

    def __repr__(self):
        return "C(x=%r)" % self.x


def main():
    grammar = generate_cfg(C)
    print(grammar)

    sampler = ModelSampler()
    x1 = grammar.sample(sampler=sampler)
    x2 = grammar.sample(sampler=sampler)
    x3 = grammar.sample(sampler=sampler)
    print(x1, x2, x3)

    print(sampler.model)
    print(sampler.updates)


if __name__ == "__main__":
    main()
```

