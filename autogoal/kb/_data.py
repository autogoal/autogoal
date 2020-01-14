from typing import List


__all__ = [
    "DataType",
    "Document",
    "Sentence",
    "Word",
    "Category",
    "Vector",
    "Matrix",
    "DenseMatrix",
    "SparseMatrix",
]


class DataType:
    @property
    def internal_type(self):
        return self._internal_type

class Word(DataType):
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
