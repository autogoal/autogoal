"""This module contains a custom implementation of the Pickle protocol to handle AutoGOAL's specific types.

You should always use our custom `Pickler` instance instead of the builtin `Pickler`.

This module provides high-level drop-in replacements for the `dump` and `load` methods.
They work with regular types:

>>> loads(dumps(42))
42
>>> loads(dumps("Hello world"))
'Hello world'

And it works with semantic types:

>>> from autogoal.kb import *
>>> loads(dumps(Word))
Word
>>> loads(dumps(Seq[Word]))
Seq[Word]

"""

import pickle
import io
from autogoal.experimental.semantics import SemanticType


class Pickler(pickle.Pickler):
    def reducer_override(self, obj):
        "Custom reducer for SemanticType and company..."
        if issubclass(obj, SemanticType):
            reduced_obj = obj._reduce()

            if reduced_obj is not None:
                return reduced_obj

        # Everything else as normal
        raise NotImplementedError


def dump(obj, fp):
    Pickler(fp).dump(obj)


def dumps(obj):
    with io.BytesIO() as fp:
        dump(obj,fp)
        return fp.getvalue()


def loads(s):
    with io.BytesIO(s) as fp:
        return load(fp)


def load(fp):
    return pickle.Unpickler(fp).load()


# Finally we run doctest.

if __name__ == "__main__":
    import doctest
    doctest.testmod()
