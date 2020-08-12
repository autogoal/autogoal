# `autogoal.contrib.nltk.RSLPStemmer`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/nltk/_generated.py#L76)
> `RSLPStemmer(self)`

A stemmer for Portuguese.

    >>> from nltk.stem import RSLPStemmer
    >>> st = RSLPStemmer()
    >>> # opening lines of Erico Verissimo's "Música ao Longe"
    >>> text = '''
    ... Clarissa risca com giz no quadro-negro a paisagem que os alunos
    ... devem copiar . Uma casinha de porta e janela , em cima duma
    ... coxilha .'''
    >>> for token in text.split():
    ...     print(st.stem(token))
    clariss risc com giz no quadro-negr a pais que os alun dev copi .
    uma cas de port e janel , em cim dum coxilh .
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `apply_rule`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/stem/rslp.py#L130)
> `apply_rule(self, word, rule_index)`

### `read_rule`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/stem/rslp.py#L67)
> `read_rule(self, filename)`

### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/nltk/_generated.py#L82)
> `run(self, input)`

### `stem`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/stem/rslp.py#L100)
> `stem(self, word)`

Strip affixes from the token and return the stem.

:param token: The token that should be stemmed.
:type token: str
