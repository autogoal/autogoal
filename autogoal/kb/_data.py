from autogoal.exceptions import InterfaceIncompatibleError

# import types
import inspect

# import pprint

from typing import Mapping
from autogoal.grammar import Symbol, Union, Empty, Subset

# from scipy.sparse.base import spmatrix
# from numpy import ndarray


# def algorithm(input_type, output_type):
#     def run_method(self, input: input_type) -> output_type:
#         pass

#     def body(ns):
#         ns["run"] = run_method

#     return types.new_class(
#         name="Algorithm[%s, %s]" % (input_type, output_type),
#         bases=(Interface,),
#         exec_body=body,
#     )


class Interface:
    @classmethod
    def is_compatible(cls, other_cls):
        own_methods = _get_annotations(cls, ignore=["generate_cfg", "is_compatible"])

        if not inspect.isclass(other_cls):
            return False

        if issubclass(other_cls, Interface):
            return False

        type_methods = _get_annotations(other_cls)
        return _compatible_annotations(own_methods, type_methods)

    @classmethod
    def generate_cfg(cls, grammar, head):
        symbol = head or Symbol(cls.__name__)
        compatible = []

        for _, other_cls in grammar.namespace.items():
            if cls.is_compatible(other_cls):
                compatible.append(other_cls)

        if not compatible:
            raise InterfaceIncompatibleError(cls.__name__)

        return Union(symbol.name, *compatible).generate_cfg(grammar, symbol)


class Distinct:
    def __init__(self, interface: Interface, exceptions=None):
        self.interface = interface
        self.exceptions = exceptions

    def generate_cfg(self, grammar, head):
        symbol = head or Symbol(self.__class__.__name__)
        compatible = []

        for _, other_cls in grammar.namespace.items():
            if other_cls in self.exceptions:
                continue

            if hasattr(other_cls, "__name__") and other_cls.__name__ in self.exceptions:
                continue

            if self.interface.is_compatible(other_cls):
                compatible.append(other_cls)

        if not compatible:
            raise ValueError(
                "Cannot find compatible implementations for <class %s>" % self.interface
            )

        return Subset(symbol.name, *compatible).generate_cfg(grammar, symbol)


# def conforms(type1, type2):
#     if inspect.isclass(type1) and inspect.isclass(type2):
#         return issubclass(type1, type2)

#     if hasattr(type1, "__conforms__") and type1.__conforms__(type2):
#         return True

#     if hasattr(type2, "__rconforms__") and type2.__rconforms__(type1):
#         return True

#     return False


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

            if not conforms(ann_if, ann_im):
                return False

        return_if = mif.return_annotation

        if return_if == inspect.Parameter.empty:
            continue

        return_im = mim.return_annotation

        if not conforms(return_im, return_if):
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


# # def make_list_wrapper(algorithm):
# #     from autogoal.kb._algorithm import _get_annotations

# #     input_type, output_type = _get_annotations(algorithm)
# #     name = f"List[{algorithm.__name__}]"

# #     def wrap_list(types):
# #         if isinstance(types, Tuple):
# #             return Tuple(*(List(t) for t in types.inner))

# #         return List(types)

# #     def init_method(self, inner: algorithm):
# #         self.inner = inner

# #     def run_method(self, input: wrap_list(input_type)) -> wrap_list(output_type):
# #         return [self.inner.run(x) for x in xs]

# #     def repr_method(self):
# #         return f"{name}(inner={repr(self.inner)})"

# #     def getattr_method(self, attr):
# #         return getattr(self.inner, attr)

# #     def body(ns):
# #         ns["__init__"] = init_method
# #         ns["run"] = run_method
# #         ns["__repr__"] = repr_method
# #         ns["__getattr__"] = getattr_method

# #     return types.new_class(name=name, bases=(), exec_body=body)


# # def build_composite_list(input_type, output_type, depth=1):
# #     def wrap(t, d):
# #         if d == 0:
# #             return t

# #         return List(wrap(t, d - 1))

# #     input_wrapper = wrap(input_type, depth)
# #     output_wrapper = wrap(output_type, depth)

# #     # name = "ListAlgorithm"  # % (input_wrapper, output_wrapper)
# #     name = "ListAlgorithm" #[%s, %s]" % (input_wrapper, output_wrapper)

# #     def init_method(self, inner: algorithm(input_type, output_type)):
# #         self.inner = inner

# #     def run_method(self, input: input_wrapper) -> output_wrapper:
# #         def wrap_run(xs, d):
# #             if d == 0:
# #                 return self.inner.run(xs)

# #             return [wrap_run(x, d - 1) for x in xs]

# #         return wrap_run(input, depth)

# #     def repr_method(self):
# #         return f"{name}(inner={repr(self.inner)})"

# #     def getattr_method(self, attr):
# #         return getattr(self.inner, attr)

# #     def reduce_method(self):
# #         return (
# #             build_composite_list_instance,
# #             (input_type, output_type, self.inner)
# #         )

# #     def body(ns):
# #         ns["__init__"] = init_method
# #         ns["run"] = run_method
# #         ns["__repr__"] = repr_method
# #         ns["__getattr__"] = getattr_method
# #         ns["__reduce__"] = reduce_method

# #     return types.new_class(name=name, bases=(), exec_body=body)


# # def build_composite_list_instance(input_type, output_type, inner_algorithm):
# #     """
# #     Build a ListAlgorithm[...] type and instantiate it directly on a given algorithm.
# #     """
# #     return build_composite_list(input_type, output_type)(inner_algorithm)


# # def build_composite_tuple(index, input_type: "Tuple", output_type: "Tuple"):
# #     """
# #     Dynamically generate a class `TupleAlgorithm` that wraps
# #     another algorithm to receive a Tuple but pass only one of the
# #     parameters to the internal algorithm.
# #     """

# #     internal_input = input_type.inner[index]
# #     internal_output = output_type.inner[index]

# #     name = "TupleAlgorithm" #[%s, %s]" % (input_type, output_type)

# #     def init_method(self, inner: algorithm(internal_input, internal_output)):
# #         self.inner = inner

# #     def run_method(self, input: input_type) -> output_type:
# #         elements = list(input)
# #         elements[index] = self.inner.run(elements[index])
# #         return tuple(elements)

# #     def repr_method(self):
# #         return f"{name}(inner={repr(self.inner)})"

# #     def getattr_method(self, attr):
# #         return getattr(self.inner, attr)

# #     def reduce_method(self):
# #         return (
# #             build_composite_tuple_instance,
# #             (index, input_type, output_type, self.inner)
# #         )

# #     def body(ns):
# #         ns["__init__"] = init_method
# #         ns["run"] = run_method
# #         ns["__repr__"] = repr_method
# #         ns["__getattr__"] = getattr_method
# #         ns["__reduce__"] = reduce_method

# #     return types.new_class(name=name, bases=(), exec_body=body)


# # def build_composite_tuple_instance(index, input_type, output_type, inner_algorithm):
# #     """
# #     Build a TupleAlgorithm[...] type and instantiate it directly on a given algorithm.
# #     """
# #     return build_composite_tuple(index, input_type, output_type)(inner_algorithm)


# # class DataType:
# #     def __init__(self, **tags):
# #         self.tags = tags

# #     def get_tag(self, tag):
# #         return self.tags.get(tag, None)

# #     def __repr__(self):
# #         # tags = ", ".join(
# #         #     f"{key}={value}"
# #         #     for key, value in sorted(self.tags.items(), key=lambda t: t[0])
# #         # )
# #         return f"{self.__class__.__name__}()" #({tags})"

# #     def __eq__(self, other):
# #         return repr(self) == repr(other)

# #     def __hash__(self):
# #         return hash(repr(self))

# #     @property
# #     def __name__(self):
# #         return self.__class__.__name__

# #     def __conforms__(self, other):
# #         return issubclass(self.__class__, other.__class__)


# # def infer_type(obj):
# #     """
# #     Attempts to automatically infer the most precise semantic type for `obj`.

# #     ##### Parameters

# #     * `obj`: Object to detect its semantic type.

# #     ##### Raises

# #     * `TypeError`: if no valid semantic type was found that matched `obj`.

# #     ##### Examples

# #     * Natural language

# #     ```python
# #     >>> infer_type("hello")
# #     Word()
# #     >>> infer_type("hello world")
# #     Sentence()
# #     >>> infer_type("Hello Word. It is raining.")
# #     Document()

# #     ```

# #     * Vectors

# #     ```
# #     >>> import numpy as np
# #     >>> infer_type(np.asarray(["A", "B", "C", "D"]))
# #     CategoricalVector()
# #     >>> infer_type(np.asarray([0.0, 1.1, 2.1, 0.2]))
# #     ContinuousVector()
# #     >>> infer_type(np.asarray([0, 1, 1, 0]))
# #     DiscreteVector()

# #     ```

# #     * Matrices

# #     ```
# #     >>> import numpy as np
# #     >>> infer_type(np.random.randn(10,10))
# #     MatrixContinuousDense()

# #     >>> import scipy.sparse as sp
# #     >>> infer_type(sp.coo_matrix((10,10)))
# #     MatrixContinuousSparse()

# #     ```
# #     """
# #     if isinstance(obj, str):
# #         if " " not in obj:
# #             return Word()

# #         if "." not in obj:
# #             return Sentence()

# #         return Document()

# #     if isinstance(obj, list):
# #         internal_types = set([infer_type(x) for x in obj])

# #         for test_type in [Document(), Sentence(), Word()]:
# #             if test_type in internal_types:
# #                 return List(test_type)

# #     if hasattr(obj, "shape"):
# #         if len(obj.shape) == 1:
# #             if isinstance(obj, ndarray):
# #                 if obj.dtype.kind == "U":
# #                     return CategoricalVector()
# #                 if obj.dtype.kind == "i":
# #                     return DiscreteVector()
# #                 if obj.dtype.kind == "f":
# #                     return ContinuousVector()
# #         if len(obj.shape) == 2:
# #             if isinstance(obj, spmatrix):
# #                 return MatrixContinuousSparse()
# #             if isinstance(obj, ndarray):
# #                 if obj.dtype.kind == "O":
# #                     return MatrixCategorical()
# #                 else:
# #                     return MatrixContinuousDense()

# #     raise TypeError("Cannot infer type for %r" % obj)


# # class Text(DataType):
# #     pass


# # class Word(Text):
# #     pass


# # class Stem(DataType):
# #     pass


# # class Sentence(Text):
# #     pass


# # class Document(Text):
# #     pass


# # class Category(DataType):
# #     pass


# # class Vector(DataType):
# #     pass


# # class Matrix(DataType):
# #     pass


# # class DenseMatrix(Matrix):
# #     pass


# # class SparseMatrix(Matrix):
# #     pass


# # class ContinuousVector(Vector):
# #     pass


# # class DiscreteVector(Vector):
# #     pass


# # class CategoricalVector(Vector):
# #     pass


# # class MatrixContinuous(Matrix):
# #     pass


# # class MatrixCategorical(Matrix):
# #     pass


# # class MatrixContinuousDense(MatrixContinuous, DenseMatrix):
# #     pass


# # class MatrixContinuousSparse(MatrixContinuous, SparseMatrix):
# #     pass


# # class Entity(DataType):
# #     pass


# # class Summary(Document):
# #     pass


# # class Sentiment(DataType):
# #     pass


# # class Synset(DataType):
# #     pass


# # class Postag(DataType):
# #     pass


# # class Chunktag(DataType):
# #     pass


# # class Tensor3(DataType):
# #     pass


# # class Tensor4(DataType):
# #     pass


# # class Flags(DataType):
# #     pass


# # class List(DataType):
# #     def __init__(self, inner):
# #         self.inner = inner
# #         # super().__init__(**inner.tags)

# #     def depth(self):
# #         if not isinstance(self.inner, List):
# #             return 1

# #         return 1 + self.inner.depth()

# #     def __conforms__(self, other):
# #         return isinstance(other, List) and conforms(self.inner, other.inner)

# #     def __repr__(self):
# #         return "List(%r)" % self.inner


# # class Tuple(DataType):
# #     def __init__(self, *inner):
# #         self.inner = inner
# #         # super().__init__(**inner[0].tags)

# #     def __repr__(self):
# #         items = ", ".join(repr(s) for s in self.inner)
# #         return "Tuple(%s)" % items

# #     def __conforms__(self, other):
# #         if not isinstance(other, Tuple):
# #             return False

# #         if len(self.inner) != len(other.inner):
# #             return False

# #         for x, y in zip(self.inner, other.inner):
# #             if not conforms(x, y):
# #                 return False

# #         return True


# # DATA_TYPES = frozenset(
# #     [
# #         Text,
# #         Word,
# #         Stem,
# #         Sentence,
# #         Document,
# #         Category,
# #         Vector,
# #         Matrix,
# #         DenseMatrix,
# #         SparseMatrix,
# #         ContinuousVector,
# #         DiscreteVector,
# #         CategoricalVector,
# #         MatrixContinuous,
# #         MatrixCategorical,
# #         MatrixContinuousDense,
# #         MatrixContinuousSparse,
# #         Entity,
# #         Summary,
# #         Sentiment,
# #         Synset,
# #         Postag,
# #         Chunktag,
# #         Tensor3,
# #         List,
# #         Tuple,
# #         Flags,
# #     ]
# # )


# # def draw_data_hierarchy(output_file):
# #     """
# #     Creates an SVG representation of the `DataType` hierarchy,
# #     for documentation purposes.
# #     """
# #     import pydot

# #     classes = list(DATA_TYPES) + [DataType]

# #     graph = pydot.Dot(direction="TB")

# #     for clss in classes:
# #         graph.add_node(pydot.Node(clss.__name__))

# #     for clss in classes:
# #         for base in clss.__bases__:
# #             if base not in classes:
# #                 continue

# #             graph.add_edge(pydot.Edge(base.__name__, clss.__name__))

# #     graph.write(output_file + ".svg", format="svg")
# #     graph.write(output_file + ".png", format="png")
