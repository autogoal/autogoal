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

    def sample(self, sampler, namespace, max_iterations):
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

    def sample(self, sampler, namespace, max_iterations):
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

    def sample(self, sampler, namespace, max_iterations):
        if max_iterations <= 0:
            raise ValueError("Max iterations exceeded")

        option = sampler.choice(self.options, handle=self.head.name)
        return self.grammar[option].sample(sampler, namespace, max_iterations-1)


class SubsetOf(Production):
    def __init__(self, head, grammar, *options, allow_empty=False):
        super().__init__(head, grammar)
        self._options: List[Symbol] = list(options)
        self._allow_empty = allow_empty

    @property
    def options(self) -> List[Symbol]:
        return self._options

    def __repr__(self):
        return "SubsetOf(options=%r)" % self._options

    def to_string(
        self, code: List[str], visited: Set[Symbol], max_symbol_length: int,
    ):
        lhs = ("<%s>" % self.head.name).ljust(max_symbol_length)
        rhs = ["<%s>" % option.name for option in self._options]

        code.append("%s := { %s }" % (lhs, " , ".join(rhs)))
        visited.add(self.head)

        for symbol in self._options:
            if symbol in visited:
                continue

            self.grammar[symbol].to_string(code, visited, max_symbol_length)

    def sample(self, sampler, namespace, max_iterations):
        if max_iterations <= 0:
            raise ValueError("Max iterations exceeded")

        while True:
            selected = []

            for option in self.options:
                if sampler.boolean(handle=self.head.name + "_" + option.name):
                    selected.append(self.grammar[option].sample(sampler, namespace, max_iterations-1))

            if len(selected) > 0 or self._allow_empty:
                break

        return selected


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

    def sample(self, sampler, namespace, max_iterations):
        if max_iterations <= 0:
            raise ValueError("Max iterations exceeded")

        kwargs = {}

        for arg, symbol in self._parameters.items():
            if isinstance(symbol, Symbol):
                arg_value = self.grammar[symbol].sample(sampler, namespace, max_iterations-1)
            else:
                arg_value = symbol

            kwargs[arg] = arg_value

        obj = namespace[self._name](**kwargs)

        if hasattr(obj, "sample") and callable(obj.sample):
            obj.sample(sampler, max_iterations=max_iterations)

        return obj


class Distribution(Callable):
    def __repr__(self):
        return "Distribution(name=%r, parameters=%r)" % (self._name, self._parameters)

    def sample(self, sampler, namespace, max_iterations):
        return sampler.distribution(self._name, handle=self.head.name, **self._parameters)


class ContextFreeGrammar(Grammar):
    """Represents a CFG grammar.
    """

    def __init__(self, start: Symbol, namespace: Dict[str, type] = None):
        super(ContextFreeGrammar, self).__init__(start)
        self._namespace = {} if namespace is None else namespace
        self._productions: Dict[Symbol, Production] = {}

    @property
    def namespace(self):
        return self._namespace

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
        return production.sample(sampler, self._namespace, max_iterations)


def generate_cfg(cls, registry=None) -> ContextFreeGrammar:
    """
    Generates a [ContextFreeGrammar](/api/autogoal.grammar/#contextfreegrammar)
    from an annotated callable (class or function).

    ##### Parameters

    * `cls`: class or function with annotated arguments.

    ##### Returns

    * `ContextFreeGrammar`: the generated grammar.

    ##### Examples

    ```python
    >>> class MyClass:
    ...     def __init__(self, x: Discrete(1,3), y: Continuous(0,1)):
    ...         pass
    >>> grammar = generate_cfg(MyClass)
    >>> print(grammar)
    <MyClass>   := MyClass (x=<MyClass_x>, y=<MyClass_y>)
    <MyClass_x> := discrete (min=1, max=3)
    <MyClass_y> := continuous (min=0, max=1)

    ```
    """
    return _generate_cfg(cls, registry=registry)


def _generate_cfg(
    cls, grammar: ContextFreeGrammar = None, head: Symbol = None, registry = None
) -> ContextFreeGrammar:
    symbol = head or Symbol(cls.__name__)

    if grammar is None:
        grammar = ContextFreeGrammar(start=symbol)

        # pre-register all classes that are already given
        if registry:
            for clss in registry:
                grammar.namespace[clss.__name__] = clss
    elif symbol in grammar:
        return grammar

    grammar.namespace[symbol.name] = cls

    if hasattr(cls, "generate_cfg"):
        return cls.generate_cfg(grammar, symbol)

    grammar.add(symbol, Empty(symbol, grammar))
    parameters = {}

    if inspect.isclass(cls):
        signature = inspect.signature(cls.__init__)
    elif inspect.isfunction(cls):
        signature = inspect.signature(cls)
    else:
        raise ValueError("Unable to obtain signature for %r" % cls)

    for param_name, param_obj in signature.parameters.items():
        if param_name in ["self", "args", "kwargs"]:
            continue

        annotation_cls = param_obj.annotation

        if annotation_cls == inspect.Parameter.empty:
            if param_obj.default == inspect.Parameter.empty:
                raise TypeError(
                    "In <%s>: Couldn't find annotation type for %r"
                    % (cls.__name__, param_obj)
                )
            continue

        if annotation_cls == "self":
            annotation_cls = cls

        if isinstance(annotation_cls, str):
            try:
                annotation_cls = grammar.namespace[annotation_cls]
            except KeyError:
                raise ValueError("To use strings for annotations, make sure recursion hits the corresponding class first.")

        if hasattr(annotation_cls, "__name__"):
            param_symbol = Symbol(annotation_cls.__name__)
        else:
            param_symbol = Symbol("%s_%s" % (cls.__name__, param_name))

        _generate_cfg(annotation_cls, grammar, param_symbol)
        parameters[param_name] = param_symbol

    grammar.replace(symbol, Callable(symbol, grammar, cls.__name__, **parameters))
    return grammar


class Discrete:
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __repr__(self):
        return "Discrete(min=%r, max=%r)" % (self.min, self.max)

    def generate_cfg(self, grammar, head):
        grammar.add(
            head, Distribution(head, grammar, "discrete", min=self.min, max=self.max)
        )
        return grammar


class Continuous(Discrete):
    def __repr__(self):
        return "Continuous(min=%r, max=%r)" % (self.min, self.max)

    def generate_cfg(self, grammar, head):
        grammar.add(
            head, Distribution(head, grammar, "continuous", min=self.min, max=self.max)
        )
        return grammar


class Categorical:
    def __init__(self, *options):
        self.options = list(options)

    def __repr__(self):
        options = ", ".join(repr(o) for o in self.options)
        return f"Categorical({options})"

    def generate_cfg(self, grammar, head):
        grammar.add(
            head, Distribution(head, grammar, "categorical", options=self.options)
        )
        return grammar


class Boolean:
    def __repr__(self):
        return f"Boolean()"

    @staticmethod
    def generate_cfg(grammar, head):
        grammar.add(head, Distribution(head, grammar, "boolean"))
        return grammar


class Union:
    def __init__(self, name, *clss):
        self.__name__ = name
        self.clss = list(clss)

    def __repr__(self):
        # args = ", ".join(str(s) for s in self.clss)
        return self.__name__

    def generate_cfg(self, grammar, head):
        symbol = head or Symbol(self.__name__)

        if symbol in grammar:
            return grammar

        grammar.add(symbol, Empty(symbol, grammar))
        children = []

        for child in self.clss:
            child_symbol = Symbol(child.__name__)
            children.append(child_symbol)
            _generate_cfg(child, grammar, child_symbol)

        grammar.replace(symbol, OneOf(symbol, grammar, *children))
        return grammar


class Subset:
    def __init__(self, name, *clss, allow_empty=False):
        self.__name__ = name
        self.clss = list(clss)
        self.allow_empty = allow_empty

    def __repr__(self):
        # args = ", ".join(str(s) for s in self.clss)
        return self.__name__

    def generate_cfg(self, grammar, head):
        symbol = head or Symbol(self.__name__)

        if symbol in grammar:
            return grammar

        grammar.add(symbol, Empty(symbol, grammar))
        children = []

        for child in self.clss:
            child_symbol = Symbol(child.__name__)
            children.append(child_symbol)
            _generate_cfg(child, grammar, child_symbol)

        grammar.replace(symbol, SubsetOf(symbol, grammar, *children, allow_empty=self.allow_empty))
        return grammar


class CfgInitializer:
    def __init__(self, registry=None):
        self._grammars = {}
        self._registry = registry

    def __call__(self, cls, sampler=None):
        if cls not in self._grammars:
            self._grammars[cls] =  generate_cfg(cls, self._registry)

        return self._grammars[cls].sample(sampler=sampler)
