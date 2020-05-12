# `autogoal.utils.compute_class_weights`

> [📝](/usr/lib/python3/dist-packages/autogoal/utils/__init__.py#L144)
> `compute_class_weights(y)`

Computes relative class weights for imbalanced datasets. Works with nested input.

##### Examples

```python
>>> compute_class_weights([['A', 'B', 'A'], ['C'], ['C', 'C']])
{'A': 1.5, 'B': 3.0, 'C': 1.0}

```
