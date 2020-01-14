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
    "List",
    "Stem"
]


class DataType:
    def __init__(self, **tags):
        self.tags = tags

    def get_tag(self, tag):
        return self.tags.get(tag, None)

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
    def __init__(self, inner_type):
        self._inner_type = inner_type
        super().__init__(**inner_type.tags)


# class Classifier:
#     def run(self, input: Document(domain='health', language='english')) -> Category():
#         pass
