# `autogoal.contrib.regex.PhoneRegex`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/regex/__init__.py#L129)
> `PhoneRegex(self, full)`

Finds if a phone number is contained inside a word using regular expressions.

##### Examples

```python
>>> regex = phoneRegex(full=True)
>>> regex.run("+619123456789")
{'is_phone_regex': True}

>>> regex = phoneRegex(full=True)
>>> regex.run("There is an phone at +619123456789, who would know?")
{'is_phone_regex': False}

>>> regex = phoneRegex(full=False)
>>> regex.run("There is an phone at +619123456789, who would know?")
{'is_phone_regex': True}

```
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/regex/__init__.py#L18)
> `run(self, input)`

