# `autogoal.utils`

## Classes

### `CacheManager`

> [📝](https://github.com/sestevez/autogoal/blob/master/autogoal/utils/_cache.py#L1)
> `CacheManager(self)`


!!! warning
    This class has no docstrings.

### `ResourceManager`

> [📝](https://github.com/sestevez/autogoal/blob/master/autogoal/utils/_resource.py#L9)
> `ResourceManager(self, time_limit=300, memory_limit=4294967296)`


Resource manager class.

##### Parameters

- `time_limit: int`: maximum amount of seconds a function can run for (default = `5 minutes`).
- `ram_limit: int`: maximum amount of memory bytes a function can use (default = `4 GB`).

##### Notes

- Only one function may be restricted at the same time.
  Upcoming updates will fix this matter.
- Memory restriction is issued upon the process's heap memory size and not
  over total address space in order to get a better estimation of the used memory.


### `RestrictedWorker`

> [📝](https://github.com/sestevez/autogoal/blob/master/autogoal/utils/_process.py#L12)
> `RestrictedWorker(self, function, timeout, memory)`


!!! warning
    This class has no docstrings.


## Functions

### `nice_repr`

> [📝](https://github.com/sestevez/autogoal/blob/master/autogoal/utils/__init__.py#L10)
> `nice_repr(cls)`


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


