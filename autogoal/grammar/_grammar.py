# coding: utf8

from typing import Mapping, List, Set
import random


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
    grammar: "Grammar" = None

    def to_string(self, head: Symbol, code: List[str], visited: Set[Symbol]):
        raise NotImplementedError()

    def sample(self, sampler, namespace):
        raise NotImplementedError()


class Empty(Production):
    def __repr__(self):
        return "Empty()"

    def to_string(self, head: Symbol, code: List[str], visited: Set[Symbol]):
        code.append("<%s> := %s" % (head.name, "<empty>"))
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

    def to_string(self, head: Symbol, code: List[str], visited: Set[Symbol]):
        lhs = "<%s>" % head.name
        rhs = ["<%s>" % option.name for option in self._options]

        code.append("%s := %s" % (lhs, " | ".join(rhs)))
        visited.add(head)

        for symbol in self._options:
            if symbol in visited:
                continue

            self.grammar[symbol].to_string(symbol, code, visited)

    def sample(self, sampler, namespace):
        option = sampler.sample(self)
        return self.grammar[option].sample(sampler, namespace)


class Callable(Production):
    def __init__(self, name, **parameters):
        self._name = name
        self._parameters = parameters

    def __repr__(self):
        return "Callable(name=%r, parameters=%r)" % (self._name, self._parameters)

    def to_string(self, head: Symbol, code: List[str], visited: Set[Symbol]):
        lhs = "<%s>" % head.name
        rhs = [
            "%s=%s" % (key, ("<%s>" % value.name) if hasattr(value, "name") else value)
            for key, value in self._parameters.items()
        ]

        code.append("%s := %s (%s)" % (lhs, self._name, ", ".join(rhs)))
        visited.add(head)

        for _, symbol in self._parameters.items():
            if symbol in visited:
                continue

            if symbol not in self.grammar:
                continue

            self.grammar[symbol].to_string(symbol, code, visited)

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
        return sampler.distribution(self, self._name, **self._parameters)


class Grammar:
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
        self[self._start_symbol].to_string(self._start_symbol, code, set())
        return "\n".join(code)

    def sample(self, sampler=None, **namespace):
        if sampler is None:
            sampler = UniformSampler()

        start_production = self[self._start_symbol]
        return start_production.sample(sampler, namespace)


class UniformSampler:
    def sample(self, production: OneOf):
        if not isinstance(production, OneOf):
            return production

        return random.choice(production.options)

    def distribution(self, dist: Distribution, name: str, min, max):
        if name == 'integer':
            return random.randint(min, max)
        elif name == 'float':
            return random.uniform(min, max)

        raise ValueError("Unrecognized distribution name: %s" % name)
