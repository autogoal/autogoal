# `autogoal.contrib.nltk.ToktokTokenizer`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/nltk/_generated.py#L431)
> `ToktokTokenizer(self)`

This is a Python port of the tok-tok.pl from
https://github.com/jonsafari/tok-tok/blob/master/tok-tok.pl

>>> toktok = ToktokTokenizer()
>>> text = u'Is 9.5 or 525,600 my favorite number?'
>>> print(toktok.tokenize(text, return_str=True))
Is 9.5 or 525,600 my favorite number ?
>>> text = u'The https://github.com/jonsafari/tok-tok/blob/master/tok-tok.pl is a website with/and/or slashes and sort of weird : things'
>>> print(toktok.tokenize(text, return_str=True))
The https://github.com/jonsafari/tok-tok/blob/master/tok-tok.pl is a website with/and/or slashes and sort of weird : things
>>> text = u'¡This, is a sentence with weird» symbols… appearing everywhere¿'
>>> expected = u'¡ This , is a sentence with weird » symbols … appearing everywhere ¿'
>>> assert toktok.tokenize(text, return_str=True) == expected
>>> toktok.tokenize(text) == [u'¡', u'This', u',', u'is', u'a', u'sentence', u'with', u'weird', u'»', u'symbols', u'…', u'appearing', u'everywhere', u'¿']
True
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/nltk/_generated.py#L437)
> `run(self, input)`

### `span_tokenize`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tokenize/api.py#L35)
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

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tokenize/toktok.py#L173)
> `tokenize(self, text, return_str=False)`

Return a tokenized copy of *s*.

:rtype: list of str
### `tokenize_sents`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tokenize/api.py#L44)
> `tokenize_sents(self, strings)`

Apply ``self.tokenize()`` to each element of ``strings``.  I.e.:

    return [self.tokenize(s) for s in strings]

:rtype: list(list(str))
