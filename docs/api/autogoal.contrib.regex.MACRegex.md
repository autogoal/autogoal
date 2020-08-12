# `autogoal.contrib.regex.MACRegex`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/regex/__init__.py#L77)
> `MACRegex(self, full)`

Finds if a MAC-address is contained inside a word using regular expressions.

##### Examples

```python
>>> regex = MACRegex(full=True)
>>> regex.run("3D:F2:C9:A6:B3:4F")
{'is_mac_regex': True}

>>> regex = MACRegex(full=True)
>>> regex.run("There is an IP at 3D-F2-C9-A6-B3-4F, who would know?")
{'is_mac_regex': False}

>>> regex = MACRegex(full=False)
>>> regex.run("There is an IP at 3D:F2:C9:A6:B3:4F, who would know?")
{'is_mac_regex': True}

```
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/regex/__init__.py#L18)
> `run(self, input)`

