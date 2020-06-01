# `autogoal.sampling.merge_updates`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/sampling/__init__.py#L558)
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
