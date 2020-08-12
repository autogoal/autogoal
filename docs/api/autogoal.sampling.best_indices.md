# `autogoal.sampling.best_indices`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/sampling/__init__.py#L515)
> `best_indices(values, k=1, maximize=False)`

Computes the `k` best indices from values, i.e., the indices of the values
that are the top minimum (or maximum).

##### Parameters

* `values: List`: Values to compare, must be a sortable type (e.g., `int`, `float`, ...).
* `k: int`: Number of indices to calculate. Defaults to `1`.
* `maximize: bool`: Whether to compute the maximum or minimum values. Defaults to `False`, i.e., minimize by default.

##### Returns:

* `indices: List[int]`: list of the indices that correspond to maximum (or minimum) values in `values`.

##### Examples:

```python
>>> best_indices([.33, 0.12, 0.55, 0.09], k=2)
[1, 3]

>>> best_indices([.33, 0.12, 0.55, 0.09], k=3, maximize=True)
[0, 1, 2]

>>> best_indices([.33, 0.12, 0.55, 0.09])
[3]

```

!!! note
    Note that indices are returned in their original order, **not** in the order in which
    the values would be sorted themselves.
