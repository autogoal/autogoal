from ._resource import ResourceManager
from .process import  RestrictedWorker


import inspect


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
    """
    init_signature = inspect.signature(cls.__init__)
    exclude_param_names = set(['self'])

    def repr_method(self):
        parameter_names = [name for name in init_signature.parameters if name not in exclude_param_names]
        parameter_values = [getattr(self, param, None) for param in parameter_names]
        args = ", ".join(f"{name}={repr(value)}" for name, value in zip(parameter_names, parameter_values) if value is not None)
        fr = f"{cls.__name__}({args})"

        try:
            import black
            return black.format_str(fr, mode=black.FileMode()).strip()
        except:
            return fr

    cls.__repr__ = repr_method
    return cls


