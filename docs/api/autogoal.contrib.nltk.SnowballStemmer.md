# `autogoal.contrib.nltk.SnowballStemmer`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/nltk/_generated.py#L90)
> `SnowballStemmer(self, language)`

Snowball Stemmer

The following languages are supported:
Arabic, Danish, Dutch, English, Finnish, French, German,
Hungarian, Italian, Norwegian, Portuguese, Romanian, Russian,
Spanish and Swedish.

The algorithm for English is documented here:

    Porter, M. "An algorithm for suffix stripping."
    Program 14.3 (1980): 130-137.

The algorithms have been developed by Martin Porter.
These stemmers are called Snowball, because Porter created
a programming language with this name for creating
new stemming algorithms. There is more information available
at http://snowball.tartarus.org/

The stemmer is invoked as shown below:

>>> from nltk.stem import SnowballStemmer
>>> print(" ".join(SnowballStemmer.languages)) # See which languages are supported
arabic danish dutch english finnish french german hungarian
italian norwegian porter portuguese romanian russian
spanish swedish
>>> stemmer = SnowballStemmer("german") # Choose a language
>>> stemmer.stem("Autobahnen") # Stem a word
'autobahn'

Invoking the stemmers that way is useful if you do not know the
language to be stemmed at runtime. Alternatively, if you already know
the language, then you can invoke the language specific stemmer directly:

>>> from nltk.stem.snowball import GermanStemmer
>>> stemmer = GermanStemmer()
>>> stemmer.stem("Autobahnen")
'autobahn'

:param language: The language whose subclass is instantiated.
:type language: str or unicode
:param ignore_stopwords: If set to True, stopwords are
                         not stemmed and returned unchanged.
                         Set to False by default.
:type ignore_stopwords: bool
:raise ValueError: If there is no stemmer for the specified
                       language, a ValueError is raised.
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/nltk/_generated.py#L115)
> `run(self, input)`

### `stem`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/stem/snowball.py#L114)
> `stem(self, token)`

Strip affixes from the token and return the stem.

:param token: The token that should be stemmed.
:type token: str
