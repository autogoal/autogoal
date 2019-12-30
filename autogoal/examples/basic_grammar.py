# coding: utf8

from autogoal.grammar import generate_grammar, Discrete, Union, Continuous


class A:
    def __init__(self, x: Discrete(0, 10)):
        self.x = x


class B:
    def __init__(self, y: Continuous(0, 1)):
        self.y = y


AorB = Union("AorB", A, B)


class C:
    def __init__(self, obj: AorB):
        self.obj = obj


def main():
    grammar = generate_grammar(AorB)
    print(grammar)


if __name__ == "__main__":
    main()
