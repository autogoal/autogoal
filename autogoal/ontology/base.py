# coding: utf8

import inspect
import pprint

from autogoal.grammar import Grammar, Callable, OneOf, Symbol, Empty, Distribution



def register_abstract_class(cls):
    cls.__children__ = []
    register_concrete_class(cls)
    return cls


def register_concrete_class(cls):
    for base in cls.__bases__:
        if base == BaseObject or base == BaseAbstract:
            continue

        if not hasattr(base, '__children__'):
            continue

        base.__children__.append(cls)

    return cls


class BaseAbstract:
    @classmethod
    def generate_grammar(cls, grammar=None, head=None):
        symbol = head or Symbol(cls.__name__)

        if grammar is None:
            grammar = Grammar(start_symbol=symbol)
        elif symbol in grammar:
            return grammar

        grammar.add(symbol, Empty())
        children = []

        for child in cls.__children__:
            child_symbol = Symbol(child.__name__)
            children.append(child_symbol)
            child.generate_grammar(grammar, child_symbol)

        grammar.replace(symbol, OneOf(*children))
        return grammar


class BaseObject(BaseAbstract):
    @classmethod
    def generate_grammar(cls, grammar=None, head=None):
        symbol = head or Symbol(cls.__name__)

        if grammar is None:
            grammar = Grammar(start_symbol=symbol)
        elif symbol in grammar:
            return grammar

        grammar.add(symbol, Empty())
        parameters = {}
        signature = inspect.signature(cls.__init__)

        for param_name, param_obj in signature.parameters.items():
            if param_name in ["self", "args", "kwargs"]:
                continue

            param_symbol = Symbol("%s_%s" % (cls.__name__, param_name))
            annotation_cls = param_obj.annotation

            if annotation_cls == 'self':
                annotation_cls = cls

            annotation_cls.generate_grammar(grammar, param_symbol)
            parameters[param_name] = param_symbol

        grammar.replace(symbol, Callable(cls.__name__, **parameters))
        return grammar


class Discrete:
    __name__ = "Discrete"

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def generate_grammar(self, grammar, head):
        grammar.add(head, Distribution('integer', min=self.min, max=self.max))
        return grammar


class Continuous(Discrete):
    __name__ = "Continuous"

    def generate_grammar(self, grammar, head):
        grammar.add(head, Distribution('float', min=self.min, max=self.max))
        return grammar


@register_abstract_class
class Algorithm(BaseAbstract):
    pass
