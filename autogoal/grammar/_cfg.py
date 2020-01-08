import inspect
import random
import warnings
import sys
from typing import List, Dict, Set

from ._base import Grammar, Sampler


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
    def __init__(self, head: Symbol, grammar: "ContextFreeGrammar"):
        self.head = head
        self.grammar = grammar

    def to_string(
        self, code: List[str], visited: Set[Symbol], max_symbol_length: int,
    ):
        raise NotImplementedError()

    def sample(self, sampler, namespace):
        raise NotImplementedError()


class Empty(Production):
    def __repr__(self):
        return "Empty()"

    def to_string(
        self, code: List[str], visited: Set[Symbol], max_symbol_length: int,
    ):
        code.append(
            "%s := %s" % (("<%s>" % self.head.name).ljust(max_symbol_length), "<empty>")
        )
        visited.add(self.head)

    def sample(self, sampler, namespace):
        return None


class OneOf(Production):
    def __init__(self, head, grammar, *options):
        super().__init__(head, grammar)
        self._options: List[Symbol] = list(options)

    @property
    def options(self) -> List[Symbol]:
        return self._options

    def __repr__(self):
        return "OneOf(options=%r)" % self._options

    def to_string(
        self, code: List[str], visited: Set[Symbol], max_symbol_length: int,
    ):
        lhs = ("<%s>" % self.head.name).ljust(max_symbol_length)
        rhs = ["<%s>" % option.name for option in self._options]

        code.append("%s := %s" % (lhs, " | ".join(rhs)))
        visited.add(self.head)

        for symbol in self._options:
            if symbol in visited:
                continue

            self.grammar[symbol].to_string(code, visited, max_symbol_length)

    def sample(self, sampler, namespace):
        option = sampler.choice(self.options, handle=self.head)
        return self.grammar[option].sample(sampler, namespace)


class Callable(Production):
    def __init__(self, head, grammar, name, **parameters):
        super().__init__(head, grammar)
        self._name = name
        self._parameters = parameters

    def __repr__(self):
        return "Callable(name=%r, parameters=%r)" % (self._name, self._parameters)

    def to_string(
        self, code: List[str], visited: Set[Symbol], max_symbol_length: int,
    ):
        lhs = ("<%s>" % self.head.name).ljust(max_symbol_length)
        rhs = [
            "%s=%s" % (key, ("<%s>" % value.name) if hasattr(value, "name") else value)
            for key, value in self._parameters.items()
        ]

        code.append("%s := %s (%s)" % (lhs, self._name, ", ".join(rhs)))
        visited.add(self.head)

        for _, symbol in self._parameters.items():
            if not isinstance(symbol, Symbol):
                continue

            if symbol in visited:
                continue

            if symbol not in self.grammar:
                continue

            self.grammar[symbol].to_string(code, visited, max_symbol_length)

    def sample(self, sampler, namespace):
        kwargs = {}

        for arg, symbol in self._parameters.items():
            if isinstance(symbol, Symbol):
                arg_value = self.grammar[symbol].sample(sampler, namespace)
            else:
                arg_value = symbol

            kwargs[arg] = arg_value

        obj = namespace[self._name](**kwargs)

        if hasattr(obj, "sample"):
            obj.sample(sampler)

        return obj


class Distribution(Callable):
    def __repr__(self):
        return "Distribution(name=%r, parameters=%r)" % (self._name, self._parameters)

    def sample(self, sampler, namespace):
        return sampler.distribution(self._name, handle=self.head, **self._parameters)


class ContextFreeGrammar(Grammar):
    """Represents a CFG grammar.
    """

    def __init__(self, start: Symbol, namespace: Dict[str, type] = None):
        super(ContextFreeGrammar, self).__init__(start)
        self._namespace = namespace or {}
        self._productions: Dict[Symbol, Production] = {}

    def add(self, symbol: Symbol, production: Production) -> None:
        if symbol in self:
            raise ValueError(
                "Cannot add more than once, call Grammar.replace() instead."
            )

        self._productions[symbol] = production

    def replace(self, symbol: Symbol, production: Production) -> None:
        if symbol not in self:
            raise ValueError("Symbol is not defined, call Grammar.add() instead.")

        self._productions[symbol] = production

    def __contains__(self, symbol: Symbol):
        return symbol in self._productions

    def __getitem__(self, symbol: Symbol) -> Production:
        return self._productions[symbol]

    def __repr__(self):
        return "Grammar(start=%r, productions=%r)" % (self._start, self._productions,)

    def __str__(self):
        code = []
        max_symbol_length = max(len(symbol.name) for symbol in self._productions) + 2
        self[self._start].to_string(code, set(), max_symbol_length)
        return "\n".join(code)

    def _sample(self, symbol, max_iterations, sampler):
        production = self[symbol]
        return production.sample(sampler, self._namespace)


def generate_cfg(
    cls, grammar: ContextFreeGrammar = None, head: Symbol = None
) -> ContextFreeGrammar:
    symbol = head or Symbol(cls.__name__)

    if grammar is None:
        grammar = ContextFreeGrammar(start=symbol)
    elif symbol in grammar:
        return grammar

    grammar._namespace[symbol.name] = cls

    if hasattr(cls, "generate_cfg"):
        return cls.generate_cfg(grammar, symbol)

    grammar.add(symbol, Empty(symbol, grammar))
    parameters = {}
    signature = inspect.signature(cls.__init__)

    for param_name, param_obj in signature.parameters.items():
        if param_name in ["self", "args", "kwargs"]:
            continue

        annotation_cls = param_obj.annotation

        if annotation_cls == inspect.Signature.empty:
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

    grammar.replace(symbol, Callable(symbol, grammar, cls.__name__, **parameters))
    return grammar


class Discrete:
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def generate_cfg(self, grammar, head):
        grammar.add(
            head, Distribution(head, grammar, "discrete", min=self.min, max=self.max)
        )
        return grammar


class Continuous(Discrete):
    def generate_cfg(self, grammar, head):
        grammar.add(
            head, Distribution(head, grammar, "continuous", min=self.min, max=self.max)
        )
        return grammar


class Categorical:
    def __init__(self, *options):
        self.options = list(options)

    def generate_cfg(self, grammar, head):
        grammar.add(
            head, Distribution(head, grammar, "categorical", options=self.options)
        )
        return grammar


class Boolean:
    def generate_cfg(self, grammar, head):
        grammar.add(head, Distribution(head, grammar, "boolean"))
        return grammar


class Union:
    def __init__(self, name, *clss):
        self.__name__ = name
        self.clss = list(clss)

    def generate_cfg(self, grammar, head):
        symbol = head or Symbol(self.__name__)

        if symbol in grammar:
            return grammar

        grammar.add(symbol, Empty(symbol, grammar))
        children = []

        for child in self.clss:
            child_symbol = Symbol(child.__name__)
            children.append(child_symbol)
            generate_cfg(child, grammar, child_symbol)

        grammar.replace(symbol, OneOf(symbol, grammar, *children))
        return grammar


class CfgInitializer:
    def __init__(self):
        self._grammars = {}

    def __call__(self, cls):
        if cls not in self._grammars:
            self._grammars[cls] = generate_cfg(cls)

        return self._grammars[cls].sample()
