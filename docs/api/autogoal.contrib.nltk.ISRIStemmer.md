# `autogoal.contrib.nltk.ISRIStemmer`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/nltk/_generated.py#L34)
> `ISRIStemmer(self)`

ISRI Arabic stemmer based on algorithm: Arabic Stemming without a root dictionary.
Information Science Research Institute. University of Nevada, Las Vegas, USA.

A few minor modifications have been made to ISRI basic algorithm.
See the source code of this module for more information.

isri.stem(token) returns Arabic root for the given token.

The ISRI Stemmer requires that all tokens have Unicode string types.
If you use Python IDLE on Arabic Windows you have to decode text first
using Arabic '1256' coding.
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `end_w5`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/stem/isri.py#L331)
> `end_w5(self, word)`

ending step (word of length five)
### `end_w6`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/stem/isri.py#L375)
> `end_w6(self, word)`

ending step (word of length six)
### `norm`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/stem/isri.py#L221)
> `norm(self, word, num=3)`

normalization:
num=1  normalize diacritics
num=2  normalize initial hamza
num=3  both 1&2
### `pre1`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/stem/isri.py#L391)
> `pre1(self, word)`

normalize short prefix
### `pre32`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/stem/isri.py#L237)
> `pre32(self, word)`

remove length three and length two prefixes in this order
### `pro_w4`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/stem/isri.py#L267)
> `pro_w4(self, word)`

process length four patterns and extract length three roots
### `pro_w53`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/stem/isri.py#L283)
> `pro_w53(self, word)`

process length five patterns and extract length three roots
### `pro_w54`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/stem/isri.py#L321)
> `pro_w54(self, word)`

process length five patterns and extract length four roots
### `pro_w6`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/stem/isri.py#L339)
> `pro_w6(self, word)`

process length six patterns and extract length three roots
### `pro_w64`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/stem/isri.py#L367)
> `pro_w64(self, word)`

process length six patterns and extract length four roots
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/nltk/_generated.py#L40)
> `run(self, input)`

### `stem`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/stem/isri.py#L184)
> `stem(self, token)`

Stemming a word token using the ISRI stemmer.
### `suf1`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/stem/isri.py#L384)
> `suf1(self, word)`

normalize short sufix
### `suf32`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/stem/isri.py#L249)
> `suf32(self, word)`

remove length three and length two suffixes in this order
### `waw`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/stem/isri.py#L261)
> `waw(self, word)`

remove connective ‘و’ if it precedes a word beginning with ‘و’ 
