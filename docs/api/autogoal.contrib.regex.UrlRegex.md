# `autogoal.contrib.regex.UrlRegex`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/regex/__init__.py#L25)
> `UrlRegex(self, full)`

Finds if a URL is contained inside a word using regular expressions.

##### Examples

```python
>>> regex = UrlRegex(full=True)
>>> regex.run("https://autogoal.gitlab.io/autogoal/contributing/#license")
{'is_url_regex': True}

>>> regex = UrlRegex(full=True)
>>> regex.run("There is a URL at https://autogoal.gitlab.io/autogoal/contributing/#license, who would know?")
{'is_url_regex': False}

>>> regex = UrlRegex(full=False)
>>> regex.run("There is a URL at https://autogoal.gitlab.io/autogoal/contributing/#license, who would know?")
{'is_url_regex': True}

```
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/regex/__init__.py#L18)
> `run(self, input)`

