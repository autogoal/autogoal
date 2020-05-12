# `autogoal.sampling.ModelSampler`

> [📝](/usr/lib/python3/dist-packages/autogoal/sampling/__init__.py#L113)
> `ModelSampler(self, model=None, **kwargs)`

A sampler that builds and uses an internal probabilistic model to generate
values with a non-uniform probability.

For the model to work, the `handler` parameter in each sampling method
must be suplied, otherwise it behaves exactly as the standard `Sampler`.
### `boolean`

> [📝](/usr/lib/python3/dist-packages/autogoal/sampling/__init__.py#L191)
> `boolean(self, handle=None)`

Returns a boolean value.

##### Examples

```python
>>> sampler = Sampler(random_state=0)
>>> [sampler.boolean() for _ in range(10)]
[False, False, True, True, False, True, False, True, True, False]

```
### `categorical`

> [📝](/usr/lib/python3/dist-packages/autogoal/sampling/__init__.py#L199)
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
### `choice`

> [📝](/usr/lib/python3/dist-packages/autogoal/sampling/__init__.py#L156)
> `choice(self, options, handle=None)`

Returns one of the options.

##### Examples

```python
>>> sampler = Sampler(random_state=0)
>>> [sampler.choice(['A', 'B', 'C']) for _ in range(5)]
['B', 'B', 'A', 'B', 'C']

```
### `continuous`

> [📝](/usr/lib/python3/dist-packages/autogoal/sampling/__init__.py#L181)
> `continuous(self, min=0, max=1, handle=None)`

Returns a continuous value between `min` and `max`.

##### Examples

```python
>>> sampler = Sampler(random_state=0)
>>> [round(sampler.continuous(0, 10), 2) for _ in range(10)]
[8.44, 7.58, 4.21, 2.59, 5.11, 4.05, 7.84, 3.03, 4.77, 5.83]

```
### `discrete`

> [📝](/usr/lib/python3/dist-packages/autogoal/sampling/__init__.py#L171)
> `discrete(self, min=0, max=10, handle=None)`

Returns a discrete value between `min` and `max`.

##### Examples

```python
>>> sampler = Sampler(random_state=0)
>>> [sampler.discrete(0, 10) for _ in range(10)]
[6, 6, 0, 4, 8, 7, 6, 4, 7, 5]

```
### `distribution`

> [📝](/usr/lib/python3/dist-packages/autogoal/sampling/__init__.py#L39)
> `distribution(self, name, handle=None, **kwargs)`

Shortcut function for generating from a distribution,
either `discrete`, `continuous`, `boolean` or `categorical`.
