# `autogoal.contrib.nltk.SExprTokenizer`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/nltk/_generated.py#L375)
> `SExprTokenizer(self, strict)`

A tokenizer that divides strings into s-expressions.
An s-expresion can be either:

  - a parenthesized expression, including any nested parenthesized
    expressions, or
  - a sequence of non-whitespace non-parenthesis characters.

For example, the string ``(a (b c)) d e (f)`` consists of four
s-expressions: ``(a (b c))``, ``d``, ``e``, and ``(f)``.

By default, the characters ``(`` and ``)`` are treated as open and
close parentheses, but alternative strings may be specified.

:param parens: A two-element sequence specifying the open and close parentheses
    that should be used to find sexprs.  This will typically be either a
    two-character string, or a list of two strings.
:type parens: str or list
:param strict: If true, then raise an exception when tokenizing an ill-formed sexpr.
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/nltk/_generated.py#L381)
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

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tokenize/sexpr.py#L89)
> `tokenize(self, text)`

Return a list of s-expressions extracted from *text*.
For example:

    >>> SExprTokenizer().tokenize('(a b (c d)) e f (g)')
    ['(a b (c d))', 'e', 'f', '(g)']

All parentheses are assumed to mark s-expressions.
(No special processing is done to exclude parentheses that occur
inside strings, or following backslash characters.)

If the given expression contains non-matching parentheses,
then the behavior of the tokenizer depends on the ``strict``
parameter to the constructor.  If ``strict`` is ``True``, then
raise a ``ValueError``.  If ``strict`` is ``False``, then any
unmatched close parentheses will be listed as their own
s-expression; and the last partial s-expression with unmatched open
parentheses will be listed as its own s-expression:

    >>> SExprTokenizer(strict=False).tokenize('c) d) e (f (g')
    ['c', ')', 'd', ')', 'e', '(f (g']

:param text: the string to be tokenized
:type text: str or iter(str)
:rtype: iter(str)
### `tokenize_sents`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tokenize/api.py#L44)
> `tokenize_sents(self, strings)`

Apply ``self.tokenize()`` to each element of ``strings``.  I.e.:

    return [self.tokenize(s) for s in strings]

:rtype: list(list(str))
