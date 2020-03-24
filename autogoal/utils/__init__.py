import enum
import inspect
import collections


MAX_REPR_DEPTH = 10

_repr_depth = [0]


def nice_repr(cls):
    """
    A decorator that adds a nice `repr(.)` to any decorated class.

    Decorate a class with `@nice_repr` to automatically generate a `__repr__()`
    method that prints the class name along with any parameters defined in the
    constructor which can be found in `dir(self)`.

    ##### Examples

    All of the parameters that you want to be printed in `repr(.)` should
    be either stored in the instance or accesible by name (e.g., as a property).

    ```python
    >>> @nice_repr
    ... class MyType:
    ...     def __init__(self, a, b, c):
    ...         self.a = a
    ...         self._b = b
    ...         self._c = c
    ...
    ...     @property
    ...     def b(self):
    ...         return self._b
    ...
    >>> x = MyType(42, b='hello', c='world')
    >>> x
    MyType(a=42, b="hello")

    ```

    It works nicely with nested objects, if all of them are `@nice_repr` decorated.

    ```python
    >>> @nice_repr
    ... class A:
    ...     def __init__(self, inner):
    ...         self.inner = inner
    >>> @nice_repr
    ... class B:
    ...     def __init__(self, value):
    ...         self.value = value
    >>> A([B(i) for i in range(10)])
    A(
        inner=[
            B(value=0),
            B(value=1),
            B(value=2),
            B(value=3),
            B(value=4),
            B(value=5),
            B(value=6),
            B(value=7),
            B(value=8),
            B(value=9),
        ]
    )

    ```

    It works with cyclic object graphs as well:

    ```python
    >>> @nice_repr
    ... class A:
    ...     def __init__(self, a:A=None):
    ...         self.a = self
    >>> A()
    A(a=A(a=A(a=A(a=A(a=A(a=A(a=A(a=A(a=A(a=A(a=A(...))))))))))))

    ```

    !!! note
        Change `autogoal.utils.MAX_REPR_DEPTH` to increase the depth level of recursive `repr`.

    """
    def repr_method(self):
        init_signature = inspect.signature(self.__init__)
        exclude_param_names = set(['self'])

        if _repr_depth[0] > MAX_REPR_DEPTH:
            return f"{self.__class__.__name__}(...)"

        _repr_depth[0] += 1

        parameter_names = [name for name in init_signature.parameters if name not in exclude_param_names]
        parameter_values = [getattr(self, param, None) for param in parameter_names]

        if hasattr(self, '__nice_repr_hook__'):
            self.__nice_repr_hook__(parameter_names, parameter_values)

        args = ", ".join(f"{name}={repr(value)}" for name, value in zip(parameter_names, parameter_values) if value is not None)
        fr = f"{self.__class__.__name__}({args})"

        _repr_depth[0] -= 1

        try:
            import black
            return black.format_str(fr, mode=black.FileMode()).strip()
        except:
            return fr

    cls.__repr__ = repr_method
    return cls


Kb = 1024
Mb = 1024 * Kb
Gb = 1024 * Mb

Sec = 1
Min = 60 * Sec
Hour = 60 * Min


def flatten(y):
    """
    Recursively flattens a list.

    ##### Examples

    ```python
    >>> flatten([[1],[2,[3]],4])
    [1, 2, 3, 4]

    ```
    """
    if isinstance(y, list):
        return [z for x in y for z in flatten(x)]
    else:
        return [y]


def compute_class_weights(y):
    """
    Computes relative class weights for imbalanced datasets. Works with nested input.

    ##### Examples

    ```python
    >>> compute_class_weights([['A', 'B', 'A'], ['C'], ['C', 'C']])
    {'A': 1.5, 'B': 3.0, 'C': 1.0}

    ```
    """
    y = flatten(y)
    class_counts = collections.Counter(y)
    _, max_class = class_counts.most_common(1)[0]

    return {k: max_class / v for k, v in class_counts.items()}


def factory(func_or_type, *args, **kwargs):
    def call():
        return func_or_type(*args, **kwargs)

    return call


from ._resource import ResourceManager
from ._process import  RestrictedWorker, RestrictedWorkerByJoin
from ._cache import CacheManager
