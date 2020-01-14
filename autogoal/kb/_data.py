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
    "Algorithm",
    "Union",
]


class Algorithm:
    def __init__(self, input, output):
        self.input = input
        self.output = output


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
        tags = ", ".join(f"{key}={value}" for key, value in sorted(self.tags.items(), key=lambda t:t[0]))
        return f"{self.__class__.__name__}({tags})"

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __hash__(self):
        return hash(repr(self))


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


class List(DataType):
    def __init__(self, inner):
        self._inner = inner
        super().__init__(**inner.tags)

    def _conforms(self, other):
        return isinstance(other, List) and self._inner.conforms(other._inner)

    def __repr__(self):
        return "List(%r)" % self._inner


class Union(DataType):
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


# class Classifier:
#     def run(self, input: Document(domain='health', language='english')) -> Category():
#         pass
