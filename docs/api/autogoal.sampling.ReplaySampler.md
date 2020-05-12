# `autogoal.sampling.ReplaySampler`

> [📝](/usr/lib/python3/dist-packages/autogoal/sampling/__init__.py#L210)
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
`replay()` returns the same instance, to enable chaining method calls.

```python
>>> sampler.replay()
<autogoal.sampling.ReplaySampler object at ...>
>>> [sampler.discrete(0,10) for _ in range(5)]
[6, 6, 0, 4, 8]
>>> [sampler.discrete(0,10) for _ in range(5)]
[7, 6, 4, 7, 5]

```

If you try to use it in a different way as originally, it will complain.

```python
>>> sampler.replay().discrete(0,5)
Traceback (most recent call last):
    ...
TypeError: Invalid invocation of `discrete` with `args=(0, 5)`, replay history says args='(0, 10)'.

>>> sampler.replay().boolean()
Traceback (most recent call last):
    ...
TypeError: Invalid invocation of `boolean`, replay history says discrete comes next.

```
### `boolean`

> [📝](/usr/lib/python3/dist-packages/autogoal/sampling/__init__.py#L388)
> `boolean(self, *args, **kwargs)`

### `categorical`

> [📝](/usr/lib/python3/dist-packages/autogoal/sampling/__init__.py#L391)
> `categorical(self, *args, **kwargs)`

### `choice`

> [📝](/usr/lib/python3/dist-packages/autogoal/sampling/__init__.py#L376)
> `choice(self, *args, **kwargs)`

### `continuous`

> [📝](/usr/lib/python3/dist-packages/autogoal/sampling/__init__.py#L385)
> `continuous(self, *args, **kwargs)`

### `discrete`

> [📝](/usr/lib/python3/dist-packages/autogoal/sampling/__init__.py#L382)
> `discrete(self, *args, **kwargs)`

### `distribution`

> [📝](/usr/lib/python3/dist-packages/autogoal/sampling/__init__.py#L379)
> `distribution(self, *args, **kwargs)`

### `load`

> [📝](/usr/lib/python3/dist-packages/autogoal/sampling/__init__.py#L344)
> `load(fp)`

Creates a `ReplaySampler` from a stream and returns it already in
replay mode.

You are responsible for opening and closing the stream yourself.

##### Examples

```python
>>> sampler = ReplaySampler(Sampler(random_state=1))
>>> [sampler.discrete(0, 10) for _ in range(10)]
[2, 9, 1, 4, 1, 7, 7, 7, 10, 6]

>>> import io
>>> fp = io.BytesIO()
>>> sampler.replay().save(fp)
>>> fp.seek(0)
0
>>> other_sampler = ReplaySampler.load(fp)
>>> [other_sampler.discrete(0, 10) for _ in range(5)]
[2, 9, 1, 4, 1]
>>> [other_sampler.discrete(0, 10) for _ in range(5)]
[7, 7, 7, 10, 6]
### `replay`

> [📝](/usr/lib/python3/dist-packages/autogoal/sampling/__init__.py#L308)
> `replay(self)`

### `save`

> [📝](/usr/lib/python3/dist-packages/autogoal/sampling/__init__.py#L313)
> `save(self, fp)`

Saves the state of a `ReplaySampler` to a stream. It must be in replay mode.

You are responsible for opening and closing the stream yourself.

##### Examples

In this example we create a sampler, and save its state into a `StringIO`
stream to be able to see what's being saved.

```python
>>> sampler = ReplaySampler(Sampler(random_state=0))
>>> [sampler.discrete(0, 10) for _ in range(3)]
[6, 6, 0]

>>> import io
>>> fp = io.BytesIO()
>>> sampler.replay().save(fp)
>>> len(fp.getvalue())
183

```
