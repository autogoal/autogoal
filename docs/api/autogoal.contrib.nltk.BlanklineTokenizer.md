# `autogoal.contrib.nltk.BlanklineTokenizer`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/nltk/_generated.py#L333)
> `BlanklineTokenizer(self)`

Tokenize a string, treating any sequence of blank lines as a delimiter.
Blank lines are defined as lines containing no characters, except for
space or tab characters.
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/nltk/_generated.py#L339)
> `run(self, input)`

### `span_tokenize`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tokenize/regexp.py#L135)
> `span_tokenize(self, text)`

Identify the tokens using integer offsets ``(start_i, end_i)``,
where ``s[start_i:end_i]`` is the corresponding token.

:rtype: iter(tuple(int, int))
### `span_tokenize_sents`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tokenize/api.py#L54)
> `span_tokenize_sents(self, strings)`

Apply ``self.span_tokenize()`` to each element of ``strings``.  I.e.:

    return [self.span_tokenize(s) for s in strings]

:rtype: iter(list(tuple(int, int)))
### `tokenize`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tokenize/regexp.py#L122)
> `tokenize(self, text)`

Return a tokenized copy of *s*.

:rtype: list of str
### `tokenize_sents`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tokenize/api.py#L44)
> `tokenize_sents(self, strings)`

Apply ``self.tokenize()`` to each element of ``strings``.  I.e.:

    return [self.tokenize(s) for s in strings]

:rtype: list(list(str))
