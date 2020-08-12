# `autogoal.contrib.nltk.Cistem`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/nltk/_generated.py#L20)
> `Cistem(self, case_insensitive)`

CISTEM Stemmer for German

This is the official Python implementation of the CISTEM stemmer.
It is based on the paper
Leonie Weissweiler, Alexander Fraser (2017). Developing a Stemmer for German
Based on a Comparative Analysis of Publicly Available Stemmers.
In Proceedings of the German Society for Computational Linguistics and Language
Technology (GSCL)
which can be read here:
http://www.cis.lmu.de/~weissweiler/cistem/

In the paper, we conducted an analysis of publicly available stemmers,
developed two gold standards for German stemming and evaluated the stemmers
based on the two gold standards. We then proposed the stemmer implemented here
and show that it achieves slightly better f-measure than the other stemmers and
is thrice as fast as the Snowball stemmer for German while being about as fast
as most other stemmers.

case_insensitive is a a boolean specifying if case-insensitive stemming
should be used. Case insensitivity improves performance only if words in the
text may be incorrectly upper case. For all-lowercase and correctly cased
text, best performance is achieved by setting case_insensitive for false.

:param case_insensitive: if True, the stemming is case insensitive. False by default.
:type case_insensitive: bool
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `replace_back`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/stem/cistem.py#L63)
> `replace_back(word)`

### `replace_to`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/stem/cistem.py#L54)
> `replace_to(word)`

### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/nltk/_generated.py#L26)
> `run(self, input)`

### `segment`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/stem/cistem.py#L139)
> `segment(self, word)`

This method works very similarly to stem (:func:'cistem.stem'). The difference is that in
addition to returning the stem, it also returns the rest that was removed at
the end. To be able to return the stem unchanged so the stem and the rest
can be concatenated to form the original word, all subsitutions that altered
the stem in any other way than by removing letters at the end were left out.

:param word: the word that is to be stemmed
:type word: unicode
:return word: the stemmed word
:rtype: unicode
:return word: the removed suffix
:rtype: unicode

>>> from nltk.stem.cistem import Cistem
>>> stemmer = Cistem()
>>> s1 = "Speicherbehältern"
>>> print("('" + stemmer.segment(s1)[0] + "', '" + stemmer.segment(s1)[1] + "')")
('speicherbehält', 'ern')
>>> s2 = "Grenzpostens"
>>> stemmer.segment(s2)
('grenzpost', 'ens')
>>> s3 = "Ausgefeiltere"
>>> stemmer.segment(s3)
('ausgefeilt', 'ere')
>>> stemmer = Cistem(True)
>>> print("('" + stemmer.segment(s1)[0] + "', '" + stemmer.segment(s1)[1] + "')")
('speicherbehäl', 'tern')
>>> stemmer.segment(s2)
('grenzpo', 'stens')
>>> stemmer.segment(s3)
('ausgefeil', 'tere')
### `stem`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/stem/cistem.py#L72)
> `stem(self, word)`

This method takes the word to be stemmed and returns the stemmed word.

:param word: the word that is to be stemmed
:type word: unicode
:return word: the stemmed word
:rtype: unicode

>>> from nltk.stem.cistem import Cistem
>>> stemmer = Cistem()
>>> s1 = "Speicherbehältern"
>>> stemmer.stem(s1)
'speicherbehalt'
>>> s2 = "Grenzpostens"
>>> stemmer.stem(s2)
'grenzpost'
>>> s3 = "Ausgefeiltere"
>>> stemmer.stem(s3)
'ausgefeilt'
>>> stemmer = Cistem(True)
>>> stemmer.stem(s1)
'speicherbehal'
>>> stemmer.stem(s2)
'grenzpo'
>>> stemmer.stem(s3)
'ausgefeil'
