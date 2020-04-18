from autogoal.grammar import Union, Subset, generate_cfg
from autogoal.utils import nice_repr

@nice_repr
class A:
    pass

@nice_repr
class B:
    pass

@nice_repr
class C:
    pass

D = Union("D", B, C)

@nice_repr
class Main:
    def __init__(self, xs: Subset("xs", A, D)):
        self.xs = xs


grammar = generate_cfg(Main)

print(grammar)
print(grammar.sample())
