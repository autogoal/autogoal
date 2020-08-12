# `autogoal.contrib.nltk.MWETokenizer`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/nltk/_generated.py#L305)
> `MWETokenizer(self)`

A tokenizer that processes tokenized text and merges multi-word expressions
into single tokens.
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `add_mwe`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tokenize/mwe.py#L58)
> `add_mwe(self, mwe)`

Add a multi-word expression to the lexicon (stored as a word trie)

We use ``util.Trie`` to represent the trie. Its form is a dict of dicts. 
The key True marks the end of a valid MWE.

:param mwe: The multi-word expression we're adding into the word trie
:type mwe: tuple(str) or list(str)

:Example:

>>> tokenizer = MWETokenizer()
>>> tokenizer.add_mwe(('a', 'b'))
>>> tokenizer.add_mwe(('a', 'b', 'c'))
>>> tokenizer.add_mwe(('a', 'x'))
>>> expected = {'a': {'x': {True: None}, 'b': {True: None, 'c': {True: None}}}}
>>> tokenizer._mwes == expected
True
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/nltk/_generated.py#L311)
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

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tokenize/mwe.py#L80)
> `tokenize(self, text)`

:param text: A list containing tokenized text
:type text: list(str)
:return: A list of the tokenized text with multi-words merged together
:rtype: list(str)

:Example:

>>> tokenizer = MWETokenizer([('hors', "d'oeuvre")], separator='+')
>>> tokenizer.tokenize("An hors d'oeuvre tonight, sir?".split())
['An', "hors+d'oeuvre", 'tonight,', 'sir?']
### `tokenize_sents`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tokenize/api.py#L44)
> `tokenize_sents(self, strings)`

Apply ``self.tokenize()`` to each element of ``strings``.  I.e.:

    return [self.tokenize(s) for s in strings]

:rtype: list(list(str))
