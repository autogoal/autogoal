# `autogoal.contrib.nltk.LineTokenizer`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/nltk/_generated.py#L389)
> `LineTokenizer(self)`

Tokenize a string into its lines, optionally discarding blank lines.
This is similar to ``s.split('\n')``.

    >>> from nltk.tokenize import LineTokenizer
    >>> s = "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\n\nThanks."
    >>> LineTokenizer(blanklines='keep').tokenize(s)
    ['Good muffins cost $3.88', 'in New York.  Please buy me',
    'two of them.', '', 'Thanks.']
    >>> # same as [l for l in s.split('\n') if l.strip()]:
    >>> LineTokenizer(blanklines='discard').tokenize(s)
    ['Good muffins cost $3.88', 'in New York.  Please buy me',
    'two of them.', 'Thanks.']

:param blanklines: Indicates how blank lines should be handled.  Valid values are:

    - ``discard``: strip blank lines out of the token list before returning it.
       A line is considered blank if it contains only whitespace characters.
    - ``keep``: leave all blank lines in the token list.
    - ``discard-eof``: if the string ends with a newline, then do not generate
       a corresponding token ``''`` after that newline.
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/nltk/_generated.py#L395)
> `run(self, input)`

### `span_tokenize`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tokenize/simple.py#L124)
> `span_tokenize(self, s)`

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

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tokenize/simple.py#L113)
> `tokenize(self, s)`

Return a tokenized copy of *s*.

:rtype: list of str
### `tokenize_sents`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tokenize/api.py#L44)
> `tokenize_sents(self, strings)`

Apply ``self.tokenize()`` to each element of ``strings``.  I.e.:

    return [self.tokenize(s) for s in strings]

:rtype: list(list(str))
