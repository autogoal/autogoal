# `autogoal.sampling`

## Classes

### `DistributionParam`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/sampling/__init__.py#L340)
> `DistributionParam(self, weights)`


!!! warning
    This class has no docstrings.

#### `update`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/sampling/__init__.py#L345)
> `update(self, alpha, updates)`


!!! warning
    This class has no docstrings.


---
### `MeanDevParam`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/sampling/__init__.py#L354)
> `MeanDevParam(self, mean, dev)`


!!! warning
    This class has no docstrings.

#### `update`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/sampling/__init__.py#L359)
> `update(self, alpha, updates)`


!!! warning
    This class has no docstrings.


---
### `ModelParam`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/sampling/__init__.py#L326)
> `ModelParam(self, *args, **kwargs)`


!!! warning
    This class has no docstrings.

#### `update`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/sampling/__init__.py#L327)
> `update(self, alpha, updates)`


!!! warning
    This class has no docstrings.


---
### `ModelSampler`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/sampling/__init__.py#L108)
> `ModelSampler(self, model=None, **kwargs)`

A sampler that builds and uses an internal probabilistic model to generate
values with a non-uniform probability.

For the model to work, the `handler` parameter in each sampling method
must be suplied, otherwise it behaves exactly as the standard `Sampler`.
#### `boolean`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/sampling/__init__.py#L186)
> `boolean(self, handle=None)`

Returns a boolean value.

##### Examples

```python
>>> sampler = Sampler(random_state=0)
>>> [sampler.boolean() for _ in range(10)]
[False, False, True, True, False, True, False, True, True, False]

```
#### `categorical`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/sampling/__init__.py#L194)
> `categorical(self, options, handle=None)`

Returns one of the options.

The difference between `choice` and `categorical` is evident in more specialized
classes of `Sampler`. In the default implementation, their behavior is exactly the same.

##### Examples

```python
>>> sampler = Sampler(random_state=0)
>>> [sampler.categorical(['A', 'B', 'C']) for _ in range(5)]
['B', 'B', 'A', 'B', 'C']

```
#### `choice`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/sampling/__init__.py#L151)
> `choice(self, options, handle=None)`

Returns one of the options.

##### Examples

```python
>>> sampler = Sampler(random_state=0)
>>> [sampler.choice(['A', 'B', 'C']) for _ in range(5)]
['B', 'B', 'A', 'B', 'C']

```
#### `continuous`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/sampling/__init__.py#L176)
> `continuous(self, min=0, max=1, handle=None)`

Returns a continuous value between `min` and `max`.

##### Examples

```python
>>> sampler = Sampler(random_state=0)
>>> [round(sampler.continuous(0, 10), 2) for _ in range(10)]
[8.44, 7.58, 4.21, 2.59, 5.11, 4.05, 7.84, 3.03, 4.77, 5.83]

```
#### `discrete`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/sampling/__init__.py#L166)
> `discrete(self, min=0, max=10, handle=None)`

Returns a discrete value between `min` and `max`.

##### Examples

```python
>>> sampler = Sampler(random_state=0)
>>> [sampler.discrete(0, 10) for _ in range(10)]
[6, 6, 0, 4, 8, 7, 6, 4, 7, 5]

```
#### `distribution`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/sampling/__init__.py#L34)
> `distribution(self, name, handle=None, **kwargs)`

Shortcut function for generating from a distribution,
either `discrete`, `continuous`, `boolean` or `categorical`.

---
### `ReplaySampler`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/sampling/__init__.py#L205)
> `ReplaySampler(self, sampler)`

A sampler that records the generated values and then can replay the
same outputs in the same order.

One of the most interesting use cases for `ReplaySampler` is in conjunction with context free
or graph grammars, for generating complex objects.
You can pass a sampler wrapped in a `ReplaySampler` during generation, and then
reuse it later for generating the same object or graph.

##### Examples

First, instantiate a `ReplaySampler` with an internal `Sampler` instance and
use it normally.

```python
>>> sampler = ReplaySampler(Sampler(random_state=0))
>>> [sampler.discrete(0,10) for _ in range(10)]
[6, 6, 0, 4, 8, 7, 6, 4, 7, 5]

```

Then call the `replay` method and reuse the same values.

```python
>>> sampler.replay()
>>> [sampler.discrete(0,10) for _ in range(5)]
[6, 6, 0, 4, 8]
>>> [sampler.discrete(0,10) for _ in range(5)]
[7, 6, 4, 7, 5]

```

If you try to use it in a different way as originally, it will complain.

```python
>>> sampler.replay()
>>> sampler.discrete(0,5)
Traceback (most recent call last):
    ...
TypeError: Invalid invocation of `discrete` with `args=(0, 5)`, replay history says args='(0, 10)'.

>>> sampler.replay()
>>> sampler.boolean()
Traceback (most recent call last):
    ...
TypeError: Invalid invocation of `boolean`, replay history says discrete comes next.

```
#### `boolean`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/sampling/__init__.py#L319)
> `boolean(self, *args, **kwargs)`


!!! warning
    This class has no docstrings.

#### `categorical`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/sampling/__init__.py#L322)
> `categorical(self, *args, **kwargs)`


!!! warning
    This class has no docstrings.

#### `choice`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/sampling/__init__.py#L307)
> `choice(self, *args, **kwargs)`


!!! warning
    This class has no docstrings.

#### `continuous`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/sampling/__init__.py#L316)
> `continuous(self, *args, **kwargs)`


!!! warning
    This class has no docstrings.

#### `discrete`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/sampling/__init__.py#L313)
> `discrete(self, *args, **kwargs)`


!!! warning
    This class has no docstrings.

#### `distribution`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/sampling/__init__.py#L310)
> `distribution(self, *args, **kwargs)`


!!! warning
    This class has no docstrings.

#### `replay`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/sampling/__init__.py#L303)
> `replay(self)`


!!! warning
    This class has no docstrings.


---
### `Sampler`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/sampling/__init__.py#L8)
> `Sampler(self, random_state=None)`

Provides methods to obtain random samples with various distributions.

Can receive a `random_state` to guarantee the same values are obtained
in two different instantiations.
#### `boolean`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/sampling/__init__.py#L74)
> `boolean(self, handle=None)`

Returns a boolean value.

##### Examples

```python
>>> sampler = Sampler(random_state=0)
>>> [sampler.boolean() for _ in range(10)]
[False, False, True, True, False, True, False, True, True, False]

```
#### `categorical`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/sampling/__init__.py#L89)
> `categorical(self, options, handle=None)`

Returns one of the options.

The difference between `choice` and `categorical` is evident in more specialized
classes of `Sampler`. In the default implementation, their behavior is exactly the same.

##### Examples

```python
>>> sampler = Sampler(random_state=0)
>>> [sampler.categorical(['A', 'B', 'C']) for _ in range(5)]
['B', 'B', 'A', 'B', 'C']

```
#### `choice`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/sampling/__init__.py#L19)
> `choice(self, options, handle=None)`

Returns one of the options.

##### Examples

```python
>>> sampler = Sampler(random_state=0)
>>> [sampler.choice(['A', 'B', 'C']) for _ in range(5)]
['B', 'B', 'A', 'B', 'C']

```
#### `continuous`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/sampling/__init__.py#L59)
> `continuous(self, min=0, max=1, handle=None)`

Returns a continuous value between `min` and `max`.

##### Examples

```python
>>> sampler = Sampler(random_state=0)
>>> [round(sampler.continuous(0, 10), 2) for _ in range(10)]
[8.44, 7.58, 4.21, 2.59, 5.11, 4.05, 7.84, 3.03, 4.77, 5.83]

```
#### `discrete`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/sampling/__init__.py#L44)
> `discrete(self, min=0, max=10, handle=None)`

Returns a discrete value between `min` and `max`.

##### Examples

```python
>>> sampler = Sampler(random_state=0)
>>> [sampler.discrete(0, 10) for _ in range(10)]
[6, 6, 0, 4, 8, 7, 6, 4, 7, 5]

```
#### `distribution`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/sampling/__init__.py#L34)
> `distribution(self, name, handle=None, **kwargs)`

Shortcut function for generating from a distribution,
either `discrete`, `continuous`, `boolean` or `categorical`.

---
### `UnormalizedWeightParam`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/sampling/__init__.py#L332)
> `UnormalizedWeightParam(self, value)`


!!! warning
    This class has no docstrings.

#### `update`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/sampling/__init__.py#L336)
> `update(self, alpha, updates)`


!!! warning
    This class has no docstrings.


---
### `WeightParam`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/sampling/__init__.py#L369)
> `WeightParam(self, value)`


!!! warning
    This class has no docstrings.

#### `update`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/sampling/__init__.py#L373)
> `update(self, alpha, updates)`


!!! warning
    This class has no docstrings.


---

## Functions

### `best_indices`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/sampling/__init__.py#L397)
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

---
### `merge_updates`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/sampling/__init__.py#L440)
> `merge_updates(*updates)`

Merges a bunch of update dicts from `ModelSampler`
into a single dictionary.

##### Parameters:

* `updates: Sequence[Dict]`: Sequence of update dictionaries obtained
  from calling `ModelSampler.updates`.

##### Returns:

* `update: Dict`: A single dictionary with the combined (appended) updates.

##### Examples:

```python
>>> up1 = {'a': [1]}
>>> up2 = {'b': [2,3]}
>>> up3 = {'a': [4]}
>>> merge_updates(up1, up2, up3)
{'a': [1, 4], 'b': [2, 3]}

```

---
### `update_model`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/sampling/__init__.py#L378)
> `update_model(model, updates, alpha=1)`


!!! warning
    This class has no docstrings.


---
