# coding: utf8

import inspect
import random
import warnings
import sys
from typing import List, Mapping, Set


class Symbol:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return "Symbol(name=%r)" % (self.name)


class Production:
    grammar: "ContextFreeGrammar" = None

    def to_string(
        self,
        head: Symbol,
        code: List[str],
        visited: Set[Symbol],
        max_symbol_length: int,
    ):
        raise NotImplementedError()

    def sample(self, sampler, namespace):
        raise NotImplementedError()


class Empty(Production):
    def __repr__(self):
        return "Empty()"

    def to_string(
        self,
        head: Symbol,
        code: List[str],
        visited: Set[Symbol],
        max_symbol_length: int,
    ):
        code.append(
            "%s := %s" % (("<%s>" % head.name).ljust(max_symbol_length), "<empty>")
        )
        visited.add(head)

    def sample(self, sampler, namespace):
        return None


class OneOf(Production):
    def __init__(self, *options):
        self._options: List[Symbol] = list(options)

    @property
    def options(self) -> List[Symbol]:
        return self._options

    def __repr__(self):
        return "OneOf(options=%r)" % self._options

    def to_string(
        self,
        head: Symbol,
        code: List[str],
        visited: Set[Symbol],
        max_symbol_length: int,
    ):
        lhs = ("<%s>" % head.name).ljust(max_symbol_length)
        rhs = ["<%s>" % option.name for option in self._options]

        code.append("%s := %s" % (lhs, " | ".join(rhs)))
        visited.add(head)

        for symbol in self._options:
            if symbol in visited:
                continue

            self.grammar[symbol].to_string(symbol, code, visited, max_symbol_length)

    def sample(self, sampler, namespace):
        option = sampler.sample(self)
        return self.grammar[option].sample(sampler, namespace)


class Callable(Production):
    def __init__(self, name, **parameters):
        self._name = name
        self._parameters = parameters

    def __repr__(self):
        return "Callable(name=%r, parameters=%r)" % (self._name, self._parameters)

    def to_string(
        self,
        head: Symbol,
        code: List[str],
        visited: Set[Symbol],
        max_symbol_length: int,
    ):
        lhs = ("<%s>" % head.name).ljust(max_symbol_length)
        rhs = [
            "%s=%s" % (key, ("<%s>" % value.name) if hasattr(value, "name") else value)
            for key, value in self._parameters.items()
        ]

        code.append("%s := %s (%s)" % (lhs, self._name, ", ".join(rhs)))
        visited.add(head)

        for _, symbol in self._parameters.items():
            if not isinstance(symbol, Symbol):
                continue

            if symbol in visited:
                continue

            if symbol not in self.grammar:
                continue

            self.grammar[symbol].to_string(symbol, code, visited, max_symbol_length)

    def sample(self, sampler, namespace):
        kwargs = {}

        for arg, symbol in self._parameters.items():

            if isinstance(symbol, Symbol):
                arg_value = self.grammar[symbol].sample(sampler, namespace)
            else:
                arg_value = symbol

            kwargs[arg] = arg_value

        return namespace[self._name](**kwargs)


class Distribution(Callable):
    def __repr__(self):
        return "Distribution(name=%r, parameters=%r)" % (self._name, self._parameters)

    def sample(self, sampler, namespace):
        return sampler.distribution(self._name, **self._parameters)


class ContextFreeGrammar:
    namespace: Mapping = {}

    def __init__(self, start_symbol: Symbol):
        self._start_symbol: Symbol = start_symbol
        self._productions: Mapping[Symbol, Production] = {}

    def add(self, symbol: Symbol, production: Production) -> None:
        if symbol in self:
            raise ValueError(
                "Cannot add more than once, call Grammar.replace() instead."
            )

        production.grammar = self
        self._productions[symbol] = production

    def replace(self, symbol: Symbol, production: Production) -> None:
        production.grammar = self
        self._productions[symbol] = production

    def __contains__(self, symbol: Symbol):
        return symbol in self._productions

    def __getitem__(self, symbol: Symbol) -> Production:
        return self._productions[symbol]

    def __repr__(self):
        return "Grammar(start_symbol=%r, productions=%r)" % (
            self._start_symbol,
            self._productions,
        )

    def __str__(self):
        code = []
        max_symbol_length = max(len(symbol.name) for symbol in self._productions) + 2
        self[self._start_symbol].to_string(
            self._start_symbol, code, set(), max_symbol_length
        )
        return "\n".join(code)

    def sample(self, sampler=None, namespace=None):
        if namespace is None:
            namespace = self.namespace

        if sampler is None:
            sampler = UniformSampler()

        start_production = self[self._start_symbol]
        return start_production.sample(sampler, namespace)


class UniformSampler:
    def sample(self, production: OneOf):
        if not isinstance(production, OneOf):
            return production

        return random.choice(production.options)

    def distribution(self, name: str, **kwargs):
        if name == "discrete":
            return random.randint(kwargs["min"], kwargs["max"])
        elif name == "continuous":
            return random.uniform(kwargs["min"], kwargs["max"])
        elif name == "boolean":
            return random.uniform(0, 1) < 0.5
        elif name == "categorical":
            return random.choice(kwargs["options"])

        raise ValueError("Unrecognized distribution name: %s" % name)


def generate_cfg(
    cls, grammar: ContextFreeGrammar = None, head: Symbol = None
) -> ContextFreeGrammar:
    symbol = head or Symbol(cls.__name__)

    if grammar is None:
        grammar = ContextFreeGrammar(start_symbol=symbol)
        grammar.namespace = vars(sys.modules[cls.__module__])
    elif symbol in grammar:
        return grammar

    if hasattr(cls, "generate_cfg"):
        return cls.generate_cfg(grammar, symbol)

    grammar.add(symbol, Empty())
    parameters = {}
    signature = inspect.signature(cls.__init__)

    for param_name, param_obj in signature.parameters.items():
        if param_name in ["self", "args", "kwargs"]:
            continue

        annotation_cls = param_obj.annotation

        if annotation_cls == inspect._empty:
            warnings.warn(
                "In <%s>: Couldn't find annotation type for %r"
                % (cls.__name__, param_obj)
            )
            continue

        if annotation_cls == "self":
            annotation_cls = cls

        if hasattr(annotation_cls, "__name__"):
            param_symbol = Symbol(annotation_cls.__name__)
        else:
            param_symbol = Symbol("%s_%s" % (cls.__name__, param_name))

        generate_cfg(annotation_cls, grammar, param_symbol)
        parameters[param_name] = param_symbol

    grammar.replace(symbol, Callable(cls.__name__, **parameters))
    return grammar


class Discrete:
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def generate_cfg(self, grammar, head):
        grammar.add(head, Distribution("discrete", min=self.min, max=self.max))
        return grammar


class Continuous(Discrete):
    def generate_cfg(self, grammar, head):
        grammar.add(head, Distribution("continuous", min=self.min, max=self.max))
        return grammar


class Categorical:
    def __init__(self, *options):
        self.options = list(options)

    def generate_cfg(self, grammar, head):
        grammar.add(head, Distribution("categorical", options=self.options))
        return grammar


class Boolean:
    def generate_cfg(self, grammar, head):
        grammar.add(head, Distribution("boolean"))
        return grammar


class Union:
    def __init__(self, name, *clss):
        self.__name__ = name
        self.clss = list(clss)

    def generate_cfg(self, grammar, head):
        symbol = head or Symbol(self.__name__)

        if symbol in grammar:
            return grammar

        grammar.add(symbol, Empty())
        children = []

        for child in self.clss:
            child_symbol = Symbol(child.__name__)
            children.append(child_symbol)
            generate_cfg(child, grammar, child_symbol)

        grammar.replace(symbol, OneOf(*children))
        return grammar
