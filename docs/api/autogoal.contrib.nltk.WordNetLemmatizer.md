# `autogoal.contrib.nltk.WordNetLemmatizer`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/nltk/_generated.py#L123)
> `WordNetLemmatizer(self)`

WordNet Lemmatizer

Lemmatize using WordNet's built-in morphy function.
Returns the input word unchanged if it cannot be found in WordNet.

    >>> from nltk.stem import WordNetLemmatizer
    >>> wnl = WordNetLemmatizer()
    >>> print(wnl.lemmatize('dogs'))
    dog
    >>> print(wnl.lemmatize('churches'))
    church
    >>> print(wnl.lemmatize('aardwolves'))
    aardwolf
    >>> print(wnl.lemmatize('abaci'))
    abacus
    >>> print(wnl.lemmatize('hardrock'))
    hardrock
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `lemmatize`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/stem/wordnet.py#L37)
> `lemmatize(self, word, pos='n')`

### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/nltk/_generated.py#L129)
> `run(self, input)`

