# `autogoal.contrib.regex.EmailRegex`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/regex/__init__.py#L103)
> `EmailRegex(self, full)`

Finds if an email is contained inside a word using regular expressions.

##### Examples

```python
>>> regex = EmailRegex(full=True)
>>> regex.run("someone@example.com")
{'is_email_regex': True}

>>> regex = EmailRegex(full=True)
>>> regex.run("There is an email at someone@example.com, who would know?")
{'is_email_regex': False}

>>> regex = EmailRegex(full=False)
>>> regex.run("There is an email at someone@example.com, who would know?")
{'is_email_regex': True}

```
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/regex/__init__.py#L18)
> `run(self, input)`

