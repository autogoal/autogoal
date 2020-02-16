# `autogoal.grammar`

## Classes

### `Block`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/grammar/_graph.py#L207)
> `Block(self, *items)`


!!! warning
    This class has no docstrings.


---
### `Boolean`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/grammar/_cfg.py#L335)
> `Boolean(self, *args, **kwargs)`


!!! warning
    This class has no docstrings.


---
### `Categorical`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/grammar/_cfg.py#L320)
> `Categorical(self, *options)`


!!! warning
    This class has no docstrings.


---
### `CfgInitializer`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/grammar/_cfg.py#L371)
> `CfgInitializer(self, registry=None)`


!!! warning
    This class has no docstrings.


---
### `ContextFreeGrammar`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/grammar/_cfg.py#L152)
> `ContextFreeGrammar(self, start, namespace=None)`

Represents a CFG grammar.


---
### `Continuous`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/grammar/_cfg.py#L309)
> `Continuous(self, min, max)`


!!! warning
    This class has no docstrings.


---
### `Discrete`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/grammar/_cfg.py#L294)
> `Discrete(self, min, max)`


!!! warning
    This class has no docstrings.


---
### `Empty`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/grammar/_cfg.py#L38)
> `Empty(self, head, grammar)`


!!! warning
    This class has no docstrings.


---
### `Epsilon`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/grammar/_graph.py#L229)
> `Epsilon(self, *args, **kwargs)`


!!! warning
    This class has no docstrings.


---
### `Grammar`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/grammar/_base.py#L30)
> `Grammar(self, start)`


!!! warning
    This class has no docstrings.


---
### `Graph`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/grammar/_graph.py#L11)
> `Graph(self, **attrs)`


!!! warning
    This class has no docstrings.


---
### `GraphGrammar`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/grammar/_graph.py#L243)
> `GraphGrammar(self, start, initializer=None, non_terminals=None)`


!!! warning
    This class has no docstrings.


---
### `GraphSpace`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/grammar/_graph.py#L314)
> `GraphSpace(self, graph, initializer=None)`


!!! warning
    This class has no docstrings.


---
### `Path`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/grammar/_graph.py#L183)
> `Path(self, *items)`


!!! warning
    This class has no docstrings.


---
### `Sampler`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/grammar/_base.py#L4)
> `Sampler(self, random_state=None)`


!!! warning
    This class has no docstrings.


---
### `Symbol`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/grammar/_cfg.py#L10)
> `Symbol(self, name)`


!!! warning
    This class has no docstrings.


---
### `Union`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/grammar/_cfg.py#L344)
> `Union(self, name, *clss)`


!!! warning
    This class has no docstrings.


---

## Functions

### `generate_cfg`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/grammar/_cfg.py#L199)
> `generate_cfg(cls, registry=None)`


Generates a [ContextFreeGrammar](/api/autogoal.grammar/#contextfreegrammar)
from an annotated callable (class or function).

##### Parameters

* `cls`: class or function with annotated arguments.

##### Returns

* `ContextFreeGrammar`: the generated grammar.

##### Examples

```python
>>> class MyClass:
...     def __init__(self, x: Discrete(1,3), y: Continuous(0,1)):
...         pass
>>> grammar = generate_cfg(MyClass)
>>> print(grammar)
<MyClass>   := MyClass (x=<MyClass_x>, y=<MyClass_y>)
<MyClass_x> := discrete (min=1, max=3)
<MyClass_y> := continuous (min=0, max=1)

```


---
