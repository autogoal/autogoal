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
            if not inspect.isclass(clss):
                continue

            if issubclass(clss, Interface):
                continue

            type_methods = _get_annotations(clss)

            if _compatible_annotations(own_methods, type_methods):
                compatible.append(clss)

        if not compatible:
            raise ValueError(
                "Cannot find compatible implementations for interface %r" % cls
            )

        return Union(symbol.name, *compatible).generate_cfg(grammar, symbol)


def conforms(type1, type2):
    if inspect.isclass(type1) and inspect.isclass(type2):
        return issubclass(type1, type2)

    if hasattr(type1, "__conforms__") and type1.__conforms__(type2):
        return True

    if hasattr(type2, "__rconforms__") and type2.__rconforms__(type1):
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

            if not conforms(ann_im, ann_if):
                return False

        return_if = mif.return_annotation

        if return_if == inspect.Parameter.empty:
            continue

        return_im = mim.return_annotation

        if not conforms(return_if, return_im):
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
    {'f': <Signature (self, input:int) -> float>}

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


def build_composite(index, input_type: 'Tuple', output_type: 'Tuple'):
    """
    Dynamically generate a class `CompositeAlgorithmXXX` that wraps
    another algorithm to receive a Tuple but pass only one of the
    parameters to the internal algorithm.
    """

    internal_input = input_type.inner[index]
    internal_output = output_type.inner[index]
    name = 'CompositeAlgorithm[%s, %s]' % (input_type, output_type)

    def init_method(self, inner: algorithm(internal_input, internal_output)):
        self.inner = inner

    def run_method(self, input: input_type) -> output_type:
        elements = list(input)
        elements[index] = self.inner.run(elements[index])
        return tuple(elements)

    def repr_method(self):
        return f"{name}(inner={repr(self.inner)})"

    def body(ns):
        ns['__init__'] = init_method
        ns['run'] = run_method
        ns['__repr__'] = repr_method

    return types.new_class(
        name=name,
        bases=(),
        exec_body=body
    )


class DataType:
    def __init__(self, **tags):
        self.tags = tags

    def get_tag(self, tag):
        return self.tags.get(tag, None)

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

    def __conforms__(self, other):
        return issubclass(self.__class__, other.__class__)


class Text(DataType):
    pass


class Word(Text):
    pass


class Stem(DataType):
    pass


class Sentence(Text):
    pass


class Document(Text):
    pass


class Category(DataType):
    pass


class Vector(DataType):
    pass


class Matrix(DataType):
    pass


class DenseMatrix(Matrix):
    pass


class SparseMatrix(Matrix):
    pass


class ContinuousVector(Vector):
    pass


class DiscreteVector(Vector):
    pass


class CategoricalVector(Vector):
    pass


class MatrixContinuous(Matrix):
    pass


class MatrixContinuousDense(MatrixContinuous, DenseMatrix):
    pass


class MatrixContinuousSparse(MatrixContinuous, SparseMatrix):
    pass


class Entity(DataType):
    pass


class Summary(Document):
    pass


class Sentiment(DataType):
    pass


class Synset(DataType):
    pass


class List(DataType):
    def __init__(self, inner):
        self.inner = inner
        # super().__init__(**inner.tags)

    def __conforms__(self, other):
        return isinstance(other, List) and conforms(self.inner, other.inner)

    def __repr__(self):
        return "List(%r)" % self.inner


class Tuple(DataType):
    def __init__(self, *inner):
        self.inner = inner
        # super().__init__(**inner[0].tags)

    def __repr__(self):
        items = ", ".join(repr(s) for s in self.inner)
        return "Tuple(%s)" % items

    def __conforms__(self, other):
        if not isinstance(other, Tuple):
            return False

        if len(self.inner) != len(other.inner):
            return False

        for x, y in zip(self.inner, other.inner):
            if not conforms(x, y):
                return False

        return True


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
    "Entity",
    "Summary",
    "Synset",
    "Text",
    "Sentiment",
]
