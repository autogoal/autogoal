# `autogoal.utils`

## Classes

### `CacheManager`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/utils/_cache.py#L87)
> `CacheManager(self)`


!!! warning
    This class has no docstrings.

#### `get`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/utils/_cache.py#L93)
> `get(name, func)`


!!! warning
    This class has no docstrings.

#### `instance`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/utils/_cache.py#L104)
> `instance()`


!!! warning
    This class has no docstrings.


---
### `ResourceManager`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/utils/_resource.py#L9)
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
#### `get_used_memory`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/utils/_resource.py#L85)
> `get_used_memory(self)`

Returns the amount of memory being used by the current process.
#### `run_restricted`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/utils/_resource.py#L100)
> `run_restricted(self, function, *args, **kwargs)`

Executes a given function with restricted amount of
CPU time and RAM memory usage.
#### `set_memory_limit`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/utils/_resource.py#L32)
> `set_memory_limit(self, limit)`

Set the memory limit for future restricted functions.

If memory limit is higher or equal than the current OS limit
then it won't be changed.
#### `set_time_limit`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/utils/_resource.py#L57)
> `set_time_limit(self, limit)`


!!! warning
    This class has no docstrings.


---
### `RestrictedWorker`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/utils/_process.py#L13)
> `RestrictedWorker(self, function, timeout, memory)`


!!! warning
    This class has no docstrings.

#### `get_used_memory`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/utils/_process.py#L67)
> `get_used_memory(self)`

returns the amount of memory being used by the current process
#### `run_restricted`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/utils/_process.py#L46)
> `run_restricted(self, *args, **kwargs)`

Executes a given function with restricted amount of
CPU time and RAM memory usage

---
### `RestrictedWorkerByJoin`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/utils/_process.py#L82)
> `RestrictedWorkerByJoin(self, function, timeout, memory)`


!!! warning
    This class has no docstrings.

#### `get_used_memory`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/utils/_process.py#L67)
> `get_used_memory(self)`

returns the amount of memory being used by the current process
#### `run_restricted`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/utils/_process.py#L102)
> `run_restricted(self, *args, **kwargs)`

Executes a given function with restricted amount of
CPU time and RAM memory usage

---

## Functions

### `compute_class_weights`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/utils/__init__.py#L141)
> `compute_class_weights(y)`

Computes relative class weights for imbalanced datasets. Works with nested input.

##### Examples

```python
>>> compute_class_weights([['A', 'B', 'A'], ['C'], ['C', 'C']])
{'A': 1.5, 'B': 3.0, 'C': 1.0}

```

---
### `factory`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/utils/__init__.py#L160)
> `factory(func_or_type, *args, **kwargs)`


!!! warning
    This class has no docstrings.


---
### `flatten`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/utils/__init__.py#L123)
> `flatten(y)`

Recursively flattens a list.

##### Examples

```python
>>> flatten([[1],[2,[3]],4])
[1, 2, 3, 4]

```

---
### `nice_repr`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/utils/__init__.py#L11)
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

---
