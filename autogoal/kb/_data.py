import types
import inspect

from typing import Mapping
from autogoal.grammar import Symbol, Union, Empty


def algorithm(input_type, output_type):
    def run_method(self, input: input_type) -> output_type:
        pass

    def body(ns):
        ns["run"] = run_method

    return types.new_class(
        name="Algorithm[%s, %s]" % (input_type, output_type),
        bases=(Interface,),
        exec_body=body,
    )


class Interface:
    @classmethod
    def generate_cfg(cls, grammar, head):
        symbol = head or Symbol(cls.__name__)

        own_methods = _get_annotations(cls, ignore=["generate_cfg"])
        compatible = []

        for _, clss in grammar.namespace.items():
            if issubclass(clss, Interface):
                continue

            type_methods = _get_annotations(clss)

            if _compatible_annotations(own_methods, type_methods):
                compatible.append(clss)

        if not compatible:
            raise ValueError(
                "Cannot find compatible implementations for interface %r" % cls
            )

        grammar = Union(symbol.name, *compatible).generate_cfg(grammar, symbol)
        return grammar


def _conforms(type1, type2):
    if inspect.isclass(type1) and inspect.isclass(type2):
        return issubclass(type1, type2)

    if hasattr(type1, "conforms") and type1.conforms(type2):
        return True

    return False


def _compatible_annotations(
    methods_if: Mapping[str, inspect.Signature],
    methods_im: Mapping[str, inspect.Signature],
):
    for name, mif in methods_if.items():
        if not name in methods_im:
            return False

        mim = methods_im[name]

        for name, param_if in mif.parameters.items():
            if not name in mim.parameters:
                return False

            param_im = mim.parameters[name]
            ann_if = param_if.annotation

            if ann_if == inspect.Parameter.empty:
                continue

            ann_im = param_im.annotation

            if not _conforms(ann_im, ann_if):
                return False

        return_if = mif.return_annotation

        if return_if == inspect.Parameter.empty:
            continue

        return_im = mim.return_annotation

        if not _conforms(return_if, return_im):
            return False

    return True


def _get_annotations(clss, ignore=[]):
    """
    Computes the annotations of all public methods in type `clss`.

    ##### Examples

    ```python
    >>> class A:
    ...     def f(self, input: int) -> float:
    ...         pass
    >>> _get_annotations(A)

    ```
    """
    methods = inspect.getmembers(
        clss, lambda m: inspect.ismethod(m) or inspect.isfunction(m)
    )
    signatures = {
        name: inspect.signature(method)
        for name, method in methods
        if not name.startswith("_")
    }

    for name in ignore:
        signatures.pop(name, None)

    return signatures


class DataType:
    def __init__(self, **tags):
        self.tags = tags

    def get_tag(self, tag):
        return self.tags.get(tag, None)

    def conforms(self, other):
        return self._conforms(other) or other._rconforms(self)

    def _conforms(self, other):
        return issubclass(self.__class__, other.__class__)

    def _rconforms(self, other):
        return issubclass(self.__class__, other.__class__)

    def __repr__(self):
        tags = ", ".join(
            f"{key}={value}"
            for key, value in sorted(self.tags.items(), key=lambda t: t[0])
        )
        return f"{self.__class__.__name__}({tags})"

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __hash__(self):
        return hash(repr(self))


class Word(DataType):
    pass


class Stem(DataType):
    pass


class Sentence(DataType):
    pass


class Document(DataType):
    pass


class Category(DataType):
    pass


class Vector(DataType):
    pass


class Matrix(DataType):
    pass


class DenseMatrix(DataType):
    pass


class SparseMatrix(DataType):
    pass


class ContinuousVector(DataType):
    pass


class DiscreteVector(DataType):
    pass


class CategoricalVector(DataType):
    pass


class MatrixContinuous(DataType):
    pass


class MatrixContinuousDense(DataType):
    pass


class MatrixContinuousSparse(DataType):
    pass


class List(DataType):
    def __init__(self, inner):
        self._inner = inner
        super().__init__(**inner.tags)

    def _conforms(self, other):
        return isinstance(other, List) and self._inner.conforms(other._inner)

    def __repr__(self):
        return "List(%r)" % self._inner


class Tuple(DataType):
    def __init__(self, *inner):
        self._inner = sorted(inner, key=repr)
        super().__init__(**inner[0].tags)

    def __repr__(self):
        items = ", ".join(repr(s) for s in self._inner)
        return "Union(%s)" % items

    def _conforms(self, other):
        if not isinstance(other, Union):
            return False

        for x in self._inner:
            for y in other._inner:
                if x._conforms(y):
                    break
            else:
                return False

        return True

    def _rconforms(self, other):
        if isinstance(other, Union):
            return False

        for x in self._inner:
            if other._conforms(x):
                return True

        return False


__all__ = [
    "algorithm",
    "CategoricalVector",
    "Category",
    "ContinuousVector",
    "DataType",
    "DenseMatrix",
    "DiscreteVector",
    "Document",
    "List",
    "Matrix",
    "MatrixContinuous",
    "MatrixContinuousDense",
    "MatrixContinuousSparse",
    "Sentence",
    "SparseMatrix",
    "Stem",
    "Tuple",
    "Vector",
    "Word",
]
