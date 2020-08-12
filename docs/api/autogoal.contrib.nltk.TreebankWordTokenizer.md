# `autogoal.contrib.nltk.TreebankWordTokenizer`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/nltk/_generated.py#L445)
> `TreebankWordTokenizer(self)`

The Treebank tokenizer uses regular expressions to tokenize text as in Penn Treebank.
This is the method that is invoked by ``word_tokenize()``.  It assumes that the
text has already been segmented into sentences, e.g. using ``sent_tokenize()``.

This tokenizer performs the following steps:

- split standard contractions, e.g. ``don't`` -> ``do n't`` and ``they'll`` -> ``they 'll``
- treat most punctuation characters as separate tokens
- split off commas and single quotes, when followed by whitespace
- separate periods that appear at the end of line

    >>> from nltk.tokenize import TreebankWordTokenizer
    >>> s = '''Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\nThanks.'''
    >>> TreebankWordTokenizer().tokenize(s)
    ['Good', 'muffins', 'cost', '$', '3.88', 'in', 'New', 'York.', 'Please', 'buy', 'me', 'two', 'of', 'them.', 'Thanks', '.']
    >>> s = "They'll save and invest more."
    >>> TreebankWordTokenizer().tokenize(s)
    ['They', "'ll", 'save', 'and', 'invest', 'more', '.']
    >>> s = "hi, my name can't hello,"
    >>> TreebankWordTokenizer().tokenize(s)
    ['hi', ',', 'my', 'name', 'ca', "n't", 'hello', ',']
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/nltk/_generated.py#L451)
> `run(self, input)`

### `span_tokenize`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tokenize/treebank.py#L136)
> `span_tokenize(self, text)`

Uses the post-hoc nltk.tokens.align_tokens to return the offset spans.

    >>> from nltk.tokenize import TreebankWordTokenizer
    >>> s = '''Good muffins cost $3.88\nin New (York).  Please (buy) me\ntwo of them.\n(Thanks).'''
    >>> expected = [(0, 4), (5, 12), (13, 17), (18, 19), (19, 23),
    ... (24, 26), (27, 30), (31, 32), (32, 36), (36, 37), (37, 38),
    ... (40, 46), (47, 48), (48, 51), (51, 52), (53, 55), (56, 59),
    ... (60, 62), (63, 68), (69, 70), (70, 76), (76, 77), (77, 78)]
    >>> list(TreebankWordTokenizer().span_tokenize(s)) == expected
    True
    >>> expected = ['Good', 'muffins', 'cost', '$', '3.88', 'in',
    ... 'New', '(', 'York', ')', '.', 'Please', '(', 'buy', ')',
    ... 'me', 'two', 'of', 'them.', '(', 'Thanks', ')', '.']
    >>> [s[start:end] for start, end in TreebankWordTokenizer().span_tokenize(s)] == expected
    True

    Additional example
    >>> from nltk.tokenize import TreebankWordTokenizer
    >>> s = '''I said, "I'd like to buy some ''good muffins" which cost $3.88\n each in New (York)."'''
    >>> expected = [(0, 1), (2, 6), (6, 7), (8, 9), (9, 10), (10, 12),
    ... (13, 17), (18, 20), (21, 24), (25, 29), (30, 32), (32, 36),
    ... (37, 44), (44, 45), (46, 51), (52, 56), (57, 58), (58, 62),
    ... (64, 68), (69, 71), (72, 75), (76, 77), (77, 81), (81, 82),
    ... (82, 83), (83, 84)]
    >>> list(TreebankWordTokenizer().span_tokenize(s)) == expected
    True
    >>> expected = ['I', 'said', ',', '"', 'I', "'d", 'like', 'to',
    ... 'buy', 'some', "''", "good", 'muffins', '"', 'which', 'cost',
    ... '$', '3.88', 'each', 'in', 'New', '(', 'York', ')', '.', '"']
    >>> [s[start:end] for start, end in TreebankWordTokenizer().span_tokenize(s)] == expected
    True
### `span_tokenize_sents`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tokenize/api.py#L54)
> `span_tokenize_sents(self, strings)`

Apply ``self.span_tokenize()`` to each element of ``strings``.  I.e.:

    return [self.span_tokenize(s) for s in strings]

:rtype: iter(list(tuple(int, int)))
### `tokenize`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tokenize/treebank.py#L99)
> `tokenize(self, text, convert_parentheses=False, return_str=False)`

Return a tokenized copy of *s*.

:rtype: list of str
### `tokenize_sents`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tokenize/api.py#L44)
> `tokenize_sents(self, strings)`

Apply ``self.tokenize()`` to each element of ``strings``.  I.e.:

    return [self.tokenize(s) for s in strings]

:rtype: list(list(str))
