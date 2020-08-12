# `autogoal.contrib.nltk.PorterStemmer`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/nltk/_generated.py#L62)
> `PorterStemmer(self)`

A word stemmer based on the Porter stemming algorithm.

    Porter, M. "An algorithm for suffix stripping."
    Program 14.3 (1980): 130-137.

See http://www.tartarus.org/~martin/PorterStemmer/ for the homepage
of the algorithm.

Martin Porter has endorsed several modifications to the Porter
algorithm since writing his original paper, and those extensions are
included in the implementations on his website. Additionally, others
have proposed further improvements to the algorithm, including NLTK
contributors. There are thus three modes that can be selected by
passing the appropriate constant to the class constructor's `mode`
attribute:

    PorterStemmer.ORIGINAL_ALGORITHM
    - Implementation that is faithful to the original paper.

      Note that Martin Porter has deprecated this version of the
      algorithm. Martin distributes implementations of the Porter
      Stemmer in many languages, hosted at:

        http://www.tartarus.org/~martin/PorterStemmer/

      and all of these implementations include his extensions. He
      strongly recommends against using the original, published
      version of the algorithm; only use this mode if you clearly
      understand why you are choosing to do so.

    PorterStemmer.MARTIN_EXTENSIONS
    - Implementation that only uses the modifications to the
      algorithm that are included in the implementations on Martin
      Porter's website. He has declared Porter frozen, so the
      behaviour of those implementations should never change.

    PorterStemmer.NLTK_EXTENSIONS (default)
    - Implementation that includes further improvements devised by
      NLTK contributors or taken from other modified implementations
      found on the web.

For the best stemming, you should use the default NLTK_EXTENSIONS
version. However, if you need to get the same results as either the
original algorithm or one of Martin Porter's hosted versions for
compatibility with an existing implementation or dataset, you can use
one of the other modes instead.
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/nltk/_generated.py#L68)
> `run(self, input)`

### `stem`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/stem/porter.py#L651)
> `stem(self, word)`

Strip affixes from the token and return the stem.

:param token: The token that should be stemmed.
:type token: str
