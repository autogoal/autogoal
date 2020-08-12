# `autogoal.contrib.regex.IPRegex`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/regex/__init__.py#L51)
> `IPRegex(self, full)`

Finds if an IP-address is contained inside a word using regular expressions.

##### Examples

```python
>>> regex = IPRegex(full=True)
>>> regex.run("192.168.18.1")
{'is_ip_regex': True}

>>> regex = IPRegex(full=True)
>>> regex.run("There is an IP at 192.168.18.1, who would know?")
{'is_ip_regex': False}

>>> regex = IPRegex(full=False)
>>> regex.run("There is an IP at 192.168.18.1, who would know?")
{'is_ip_regex': True}

```
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/regex/__init__.py#L18)
> `run(self, input)`

