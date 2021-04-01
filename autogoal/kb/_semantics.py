"""Defines the semantic types hierarchy and utilities to infer types and work with them.
"""

# The AutoML features in AutoGOAL rest in the ability of automatically discovering pipelines
# of algorithms that are semantically annotated w.r.t. their inputs and outputs.
# This module defines the semantic types hierarchy.
#
# A semantic type is basically a class that provides a meaningful
# definition for a value in the problem domain. For example, a list of strings can either
# be a list of sentences, words, DNA sequences, or full-fledged documents. Depending on the
# interpretation of that value, we want AutoGOAL to select different algorithms, i.e., it makes
# no sense to tokenize words, while it is almost mandatory to do so when you have natural language text.

# We will never really instantiate these classes, just use them for annotations.

from functools import reduce
import inspect
import copyreg
from typing import Type


# We start by defining the base class of our hierarchy.
# A `SemanticType` is just a class that knows how to do one thing: determine if a given object
# matches its semantic definition.
# We will define a metaclass to allow for `isinstance` and `issubclass` to work based on our semantic match definition,
# and to leave space for implementing class-level `__getitem__` to allow for a generics kind-of notation.


class SemanticTypeMeta(type):
    def __instancecheck__(cls, instance) -> bool:
        return cls._match(instance)

    def __getitem__(cls, args):
        if isinstance(args, tuple):
            return cls._specialize(*args)
        else:
            return cls._specialize(args)

    def __subclasscheck__(cls, subclass: type) -> bool:
        if hasattr(subclass, "_conforms") and subclass._conforms(cls):
            return True

        return super().__subclasscheck__(subclass)

    def __call__(self, *args, **kwds):
        raise TypeError("Cannot instantiate a semantic type")

    def __repr__(cls) -> str:
        return cls._name()


# `SemanticType` defines a `match` method that implements our `isinstance` method.


class SemanticType(metaclass=SemanticTypeMeta):
    @classmethod
    def _match(cls, x) -> bool:
        return False

    @classmethod
    def _conforms(cls, other: type) -> bool:
        return False

    @classmethod
    def _name(cls) -> str:
        return cls.__name__

    @classmethod
    def _specialize(cls, *args):
        raise TypeError(f"{cls} cannot be specialized")

    @classmethod
    def _reduce(cls):
        # By default, we return the repr name of the class that
        # allows to serialize globally defined classes.
        return cls._name()

    @staticmethod
    def infer(x):
        """Automatically determines the semantic type of a given value.
        
        >>> SemanticType.infer("word")
        Word
        >>> SemanticType.infer("hello world")
        Sentence
        >>> SemanticType.infer("Hello world. This a two sentence Document.")
        Document

        It works with some tensorial types as well.

        >>> import numpy as np
        >>> SemanticType.infer(np.ones(shape=(2,2)))
        Tensor[2, Continuous, Dense]

        """
        types = inspect.getmembers(inspect.getmodule(SemanticType), inspect.isclass)
        best_type = SemanticType

        for _, t in types:
            if isinstance(x, t) and issubclass(t, best_type):
                best_type = t

        if best_type == SemanticType:
            raise ValueError(f"Cannot infer semantic type for {x}")

        return best_type


# To be able to serialize these types, we have to register a reduce function for `SemanticTypeMeta`.
# This reduce function will just dispatch to the proper instance method


def _reduce_semantic_type(t):
    return t._reduce()


copyreg.pickle(SemanticTypeMeta, _reduce_semantic_type)


# Let's start with the natural language hierarchy.


class Text(SemanticType):
    @classmethod
    def _match(cls, x):
        return isinstance(x, str)


class Document(Text):
    pass


class Sentence(Document):
    @classmethod
    def _match(cls, x):
        return super()._match(x) and x.count(".") <= 1


class Word(Sentence):
    @classmethod
    def _match(cls, x):
        return super()._match(x) and " " not in x


# We also need some basic types for generic kinds of labels, used in NLP, for example.
# Keep in mind these do not implement `_match` as there is no sensible structural way to define them.
# They are used solely to annotate algorithms.


class Label(SemanticType):
    pass


class Postag(Label):
    pass


class Chunktag(Label):
    pass


class Stem(Text):
    pass


class Synset(SemanticType):
    pass


class Sentiment(Label):
    pass


# Another basic type is a feature set, which for us is basically a dictionary of strings vs anything:


class FeatureSet(SemanticType):
    @classmethod
    def _match(self, x):
        # TODO: This is a very naive implementation of `match`.
        return isinstance(x, dict) and isinstance(list(x.keys()[0]), str)


# A first complex type we can implement is `Seq`, to represent a list (or sequence) of another semantic type.
# We want this type to be able to specialize in this notation:
#
# >>> Seq[Word]
#
# For this we have to implement `_specialize` and synthethize a new type with the corresponding
# semantic type that does the match internally.
# The challenge here is that we want to return the same `Seq` type every time we specialize on the same internal type,
# so we'll keep a class-level dictionary to store these classes as they are synthethized.
#
# Finally, we want `Seq[Word]`` to be a subclass of `Seq[Sentence]`, or more generally, `Seq[X]` to be a subclass
# of `Seq[Y]` whenever `X < Y`.


class Seq(SemanticType):
    """Represents a sequence that can be specialized in concrete internal semantic types.

    >>> isinstance(["hello", "world"], Seq[Word])
    True

    Specialized classes are exactly the same, by identity:

    >>> id(Seq[Word]) == id(Seq[Word])
    True

    Specialized classes are subclasses of the raw `Seq` class:

    >>> issubclass(Seq[Word], Seq)
    True

    And they are subclasses of other specialized classes when the internal types hold the same relationship.

    >>> issubclass(Seq[Word], Seq[Text])
    True
    >>> issubclass(Seq[Text], Seq[Word])
    False

    They are pretty-printed as well

    >>> Seq
    Seq
    >>> Seq[Word]
    Seq[Word]

    Subclasses are also serializable (which requires some non-trivial dark magic on the implementation side):

    >>> from pickle import dumps, loads
    >>> loads(dumps(Seq[Word]))
    Seq[Word]
   
    """

    __internal_types = {}

    @classmethod
    def _specialize(cls, internal_type: Type[SemanticType]):
        try:
            return Seq.__internal_types[internal_type]
        except KeyError:
            pass

        class SeqImp(Seq):
            __internal_type__ = internal_type

            @classmethod
            def _name(cls):
                return f"Seq[{internal_type}]"

            @classmethod
            def _match(cls, x):
                return isinstance(x, (list, tuple)) and internal_type._match(x[0])

            @classmethod
            def _conforms(cls, other):
                if not issubclass(other, Seq):
                    return False

                if other == Seq:
                    return True

                return issubclass(cls.__internal_type__, other.__internal_type__)

            @classmethod
            def _specialize(cls, *args, **kwargs):
                raise TypeError("Cannot specialize more a `Seq` type.")

            @classmethod
            def _reduce(cls):
                return Seq._specialize, (internal_type,)

        Seq.__internal_types[internal_type] = SeqImp

        return SeqImp


# Now let's move to the algebraic types, vectors, matrices, and tensors.
# These wrap numpy arrays of different dimensionalities
# We'll have three different semantic labels for each tensor: dimensionality (an integer),
# internal type, and a dense/sparse flag.


from numpy import ndarray
from scipy.sparse.base import spmatrix


# These instances represent the two types of tensorial structure.


class TensorStructure:
    def __init__(self, base_class: type, name: str) -> None:
        self.base_class = base_class
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def match(self, x):
        return isinstance(x, self.base_class)

    def __eq__(self, o: object) -> bool:
        return isinstance(o, TensorStructure) and o.name == self.name

    def __hash__(self) -> int:
        return hash(self.name)


Dense = TensorStructure(ndarray, "Dense")
Sparse = TensorStructure(spmatrix, "Sparse")


class TensorData:
    def __init__(self, dtype_label: str, name: str) -> None:
        self.dtype_label = dtype_label
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def match(self, x):
        return x.dtype.kind == self.dtype_label

    def __eq__(self, o: object) -> bool:
        return isinstance(o, TensorData) and o.name == self.name

    def __hash__(self) -> int:
        return hash(self.name)


Categorical = TensorData("U", "Categorical")
Continuous = TensorData("f", "Continuous")
Discrete = TensorData("i", "Discrete")


# We want the abstract `Tensor` type to be specializable using the notation:
# `Tensor[3, Category, Dense]`.
# This requires us to implement the `_specialize` just like we did with `Seq`.
# The internal class will in turn implement `_match` accordingly to how those values are defined.

# One special thing we want to do is to let some of these semantic flags undefined (using `None`)
# such that `Tensor[2, None, Dense]` represents any structure that can have two dimensions no matter the internal type.
# And we want `issubclass(...)` to work in a way that `Tensor[2, Categorical, Dense]` is a subclass to `Tensor[2, None, Dense]`.
# For this purpose we will redefine `_conforms` to match according to how those semantic flags are defined.


class Tensor(SemanticType):
    """Represents an abstract tensor type. Can be specialized into more concrete types.

    They support type inference from numpy:

    >>> import numpy as np
    >>> a_matrix = np.ones(shape=(2,2))
    >>> isinstance(a_matrix, Tensor)
    True
    >>> isinstance(a_matrix, Tensor[1, None, None])
    False
    >>> isinstance(a_matrix, Tensor[2, None, None])
    True
    >>> isinstance(a_matrix, Tensor[2, Continuous, Dense])
    True
    >>> isinstance(a_matrix, Tensor[2, Continuous, Sparse])
    False

    Tensor types also respect a special definition of subclass in which more specialized
    classes are defined as subclasses of less specialized counterparts.

    >>> issubclass(Tensor[2, Continuous, Sparse], Tensor[2, None, None])
    True
    >>> issubclass(Tensor[2, Continuous, Sparse], Tensor[2, Continuous, None])
    True
    >>> issubclass(Tensor[2, Continuous, Sparse], Tensor[2, None, Dense])
    False

    Each specific specialization is a singleton class:
    
    >>> id(Tensor[2, Continuous, Dense]) == id(Tensor[2, Continuous, Dense])
    True

    Tensor types are also serializable:

    >>> from pickle import loads, dumps
    >>> loads(dumps(Tensor[2, Continuous, Dense]))
    Tensor[2, Continuous, Dense]

    """

    __internal_types = {}

    @classmethod
    def _match(self, x):
        return isinstance(x, (ndarray, spmatrix))

    @classmethod
    def _specialize(
        cls, dimension: int, internal_type: TensorData, structure: TensorStructure
    ):
        try:
            return Tensor.__internal_types[(dimension, internal_type, structure)]
        except KeyError:
            pass

        class TensorImp(Tensor):
            __flags = (dimension, internal_type, structure)

            @classmethod
            def _name(cls):
                return f"Tensor[{cls.__flags[0]}, {cls.__flags[1]}, {cls.__flags[2]}]"

            @classmethod
            def _match(cls, x):
                if not super()._match(x):
                    return False

                if dimension is not None and (len(x.shape) != dimension):
                    return False

                if internal_type is not None and not internal_type.match(x):
                    return False

                if structure is not None and not structure.match(x):
                    return False

                return True

            @classmethod
            def _conforms(cls, other: type) -> bool:
                if other == Tensor:
                    return True

                if not hasattr(other, "_TensorImp__flags"):
                    return False

                for fmine, fother in zip(cls.__flags, other.__flags):
                    if fother is None:
                        continue

                    if fmine is None:
                        return False

                    if fmine != fother:
                        return False

                return True

            @classmethod
            def _reduce(cls):
                # To reduce Tensor implementations for pickling
                return Tensor._specialize, (dimension, internal_type, structure)

        Tensor.__internal_types[(dimension, internal_type, structure)] = TensorImp

        return TensorImp


# Now that we have the basic tensorial type implemented, we can add some aliases here.
# These aliases mostly serve for `SemanticType.infer` to work, and also to simplify imports,
# but keep in mind that anywhere we use `Vector` we just as well use `Tensor[1, None, None]`,
# as they are *exactly* the same class.

Vector = Tensor[1, None, None]
VectorContinuous = Tensor[1, Continuous, None]
VectorCategorical = Tensor[1, Categorical, Dense]
VectorDiscrete = Tensor[1, Discrete, Dense]

Matrix = Tensor[2, None, None]
MatrixContinuous = Tensor[2, Continuous, None]
MatrixContinuousDense = Tensor[2, Continuous, Dense]
MatrixContinuousSparse = Tensor[2, Continuous, Sparse]
MatrixCategorical = Tensor[2, Categorical, Dense]
MatrixDiscrete = Tensor[2, Discrete, Dense]

# 📝 Makes no sense to have sparse discrete or categorical tensors as you cannot have missing categories.

Tensor3 = Tensor[3, Continuous, Dense]
Tensor4 = Tensor[4, Continuous, Dense]

# Finally we define the publicly export classes

__all__ = [
    "SemanticType",
    "Seq",
    "Text",
    "Document",
    "Sentence",
    "Word",
    "Label",
    "Postag",
    "Chunktag",
    "Stem",
    "Synset",
    "Sentiment",
    "FeatureSet",
    "Tensor",
    "Vector",
    "VectorContinuous",
    "VectorCategorical",
    "VectorDiscrete",
    "Matrix",
    "MatrixContinuous",
    "MatrixContinuousDense",
    "MatrixContinuousSparse",
    "MatrixCategorical",
    "MatrixDiscrete",
    "Tensor3",
    "Tensor4",
    "Dense",
    "Sparse",
    "Categorical",
    "Continuous",
    "Discrete",
]


if __name__ == "__main__":
    import doctest

    doctest.testmod()
