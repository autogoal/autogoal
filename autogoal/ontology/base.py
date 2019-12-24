# coding: utf8

import inspect
import pprint


def register_abstract_class(cls):
    cls.__children__ = []
    return cls


def register_concrete_class(cls):
    for base in cls.__bases__:
        if base == BaseObject or base == BaseAbstract:
            continue

        base.__children__.append(cls)

    return cls


class BaseAbstract:
    @classmethod
    def generate_grammar(cls, grammar={}, head=None):
        lhs = head or "<%s>" % cls.__name__

        if lhs in grammar:
            return grammar

        grammar[lhs] = []
        children = []

        for inh in cls.__children__:
            children.append("<%s>" % inh.__name__)
            inh.generate_grammar(grammar)

        rhs = children

        grammar[lhs] = rhs
        return grammar


class BaseObject(BaseAbstract):
    @classmethod
    def generate_grammar(cls, grammar={}, head=None):
        lhs = head or "<%s>" % cls.__name__

        if lhs in grammar:
            return grammar

        grammar[lhs] = []
        parameters = []
        signature = inspect.signature(cls.__init__)

        for param_name, param_obj in signature.parameters.items():
            if param_name in ["self", "args", "kwargs"]:
                continue

            param_name = "<%s_%s>" % (cls.__name__, param_name)

            param_obj.annotation.generate_grammar(grammar, param_name)
            parameters.append(param_name)

        rhs = ["%s( %s )" % (cls.__name__, " ".join(parameters),)]

        grammar[lhs] = rhs
        return grammar


class Discrete:
    __name__ = "Discrete"

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def generate_grammar(self, grammar, head):
        grammar[head] = "i(%i, %i)" % (self.min, self.max)
        return grammar


class Continuous(Discrete):
    __name__ = "Continuous"

    def generate_grammar(self, grammar, head):
        grammar[head] = "f(%f, %f)" % (self.min, self.max)
        return grammar


@register_abstract_class
class Algorithm(BaseAbstract):
    pass


@register_abstract_class
class Layer(BaseAbstract):
    pass


@register_concrete_class
class SequentialLayer(BaseObject, Layer):
    def __init__(self, layer1: Layer, layer2: Layer):
        pass


@register_concrete_class
class ParallelLayer(BaseObject, Layer):
    def __init__(self, layer1: Layer, layer2: Layer):
        pass


@register_concrete_class
class DenseLayer(BaseObject, Layer):
    def __init__(self, units: Discrete(32, 2048)):
        pass


@register_concrete_class
class NeuralNetwork(BaseObject, Algorithm):
    def __init__(self, layer: Layer):
        self.layer = layer


def main():
    grammar = Algorithm.generate_grammar()

    pprint.pprint(grammar)


if __name__ == "__main__":
    main()
