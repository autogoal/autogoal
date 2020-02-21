# `autogoal.grammar.generate_cfg`

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
