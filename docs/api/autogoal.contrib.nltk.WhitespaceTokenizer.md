# `autogoal.contrib.nltk.WhitespaceTokenizer`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/nltk/_generated.py#L347)
> `WhitespaceTokenizer(self)`

Tokenize a string on whitespace (space, tab, newline).
In general, users should use the string ``split()`` method instead.

    >>> from nltk.tokenize import WhitespaceTokenizer
    >>> s = "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\n\nThanks."
    >>> WhitespaceTokenizer().tokenize(s)
    ['Good', 'muffins', 'cost', '$3.88', 'in', 'New', 'York.',
    'Please', 'buy', 'me', 'two', 'of', 'them.', 'Thanks.']
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/nltk/_generated.py#L353)
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
