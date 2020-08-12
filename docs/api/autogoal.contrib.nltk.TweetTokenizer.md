# `autogoal.contrib.nltk.TweetTokenizer`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/nltk/_generated.py#L282)
> `TweetTokenizer(self, preserve_case, reduce_len, strip_handles)`

Tokenizer for tweets.

    >>> from nltk.tokenize import TweetTokenizer
    >>> tknzr = TweetTokenizer()
    >>> s0 = "This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--"
    >>> tknzr.tokenize(s0)
    ['This', 'is', 'a', 'cooool', '#dummysmiley', ':', ':-)', ':-P', '<3', 'and', 'some', 'arrows', '<', '>', '->', '<--']

Examples using `strip_handles` and `reduce_len parameters`:

    >>> tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
    >>> s1 = '@remy: This is waaaaayyyy too much for you!!!!!!'
    >>> tknzr.tokenize(s1)
    [':', 'This', 'is', 'waaayyy', 'too', 'much', 'for', 'you', '!', '!', '!']
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/nltk/_generated.py#L297)
> `run(self, input)`

### `tokenize`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tokenize/casual.py#L278)
> `tokenize(self, text)`

:param text: str
:rtype: list(str)
:return: a tokenized list of strings; concatenating this list returns        the original string if `preserve_case=False`
