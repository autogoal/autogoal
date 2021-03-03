"""Defines the semantic types hierarchy and utilities to infer types and work with them.
"""

# The AutoML features in AutoGOAL rest in the ability of automatically discovering pipelines
# of algorithms that are semantically annotated w.r.t. their inputs and outputs.
# This module defines the semantic types hierarchy.
# 
# A semantic type is basically a class that wraps another value, and provides a meaningful
# definition for that value in the problem domain. For example, a list of strings can either
# be a list of sentences, words, DNA sequences, or full-fledged documents. Depending on the
# interpretation of that value, we want AutoGOAL to select different algorithms, i.e., it makes
# no sense to tokenize words, while it is almost mandatory to do so when you have natural language text.


import inspect
from abc import ABC, abstractclassmethod

from typing import Type


# We start by defining the base class of our hierarchy.
# A `SemanticType` is just a class that knows how to do one thing: determine if a given object
# matches its semantic definition.


class SemanticTypeMeta(type):
    def __instancecheck__(self, instance) -> bool:
        return super().__instancecheck__(instance)

    def __subclasscheck__(self, subclass: type) -> bool:
        return super().__subclasscheck__(subclass)


class SemanticType(metaclass=SemanticTypeMeta):
    def match(cls, x) -> bool:
        raise NotImplementedError

    @staticmethod
    def infer(x):
        types = inspect.getmembers(inspect.getmodule(SemanticType), 
                                   lambda m: issubclass(m, SemanticType))
        
        for _, t in types:
            if isinstance(x, t):
                return t

        raise ValueError(f"Cannot infer semantic type for {x}")


# Now we can start defining a bunch of basic semantic types. 
# Let's start with the natural language hierarchy.


class Text(SemanticType):
    @classmethod
    def match(cls, x):
        return isinstance(x, str)


class Seq(SemanticType):
    def __class_getitem__(cls, internal_type: Type[SemanticType]):
        if not issubclass(internal_type, SemanticType):
            raise ValueError("Cannot specialize on non-semantic types.")

        class SeqOf(Seq):
            @classmethod
            def match(cls, x):
                return isinstance(x, list) and internal_type.match(x[0])

            def __class_getitem__(cls, *args, **kwargs):
                raise TypeError("Cannot specialize more a `Seq` type.")

        return SeqOf


