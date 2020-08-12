# `autogoal.contrib.nltk.PunktSentenceTokenizer`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/nltk/_generated.py#L319)
> `PunktSentenceTokenizer(self)`

A sentence tokenizer which uses an unsupervised algorithm to build
a model for abbreviation words, collocations, and words that start
sentences; and then uses that model to find sentence boundaries.
This approach has been shown to work well for many European
languages.
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `debug_decisions`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tokenize/punkt.py#L1274)
> `debug_decisions(self, text)`

Classifies candidate periods as sentence breaks, yielding a dict for
each that may be used to understand why the decision was made.

See format_debug_decision() to help make this output readable.
### `dump`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tokenize/punkt.py#L1485)
> `dump(self, tokens)`

### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/nltk/_generated.py#L325)
> `run(self, input)`

### `sentences_from_text`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tokenize/punkt.py#L1319)
> `sentences_from_text(self, text, realign_boundaries=True)`

Given a text, generates the sentences in that text by only
testing candidate sentence breaks. If realign_boundaries is
True, includes in the sentence closing punctuation that
follows the period.
### `sentences_from_text_legacy`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tokenize/punkt.py#L1385)
> `sentences_from_text_legacy(self, text)`

Given a text, generates the sentences in that text. Annotates all
tokens, rather than just those with possible sentence breaks. Should
produce the same results as ``sentences_from_text``.
### `sentences_from_tokens`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tokenize/punkt.py#L1394)
> `sentences_from_tokens(self, tokens)`

Given a sequence of tokens, generates lists of tokens, each list
corresponding to a sentence.
### `span_tokenize`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tokenize/punkt.py#L1308)
> `span_tokenize(self, text, realign_boundaries=True)`

Given a text, generates (start, end) spans of sentences
in the text.
### `span_tokenize_sents`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tokenize/api.py#L54)
> `span_tokenize_sents(self, strings)`

Apply ``self.span_tokenize()`` to each element of ``strings``.  I.e.:

    return [self.span_tokenize(s) for s in strings]

:rtype: iter(list(tuple(int, int)))
### `text_contains_sentbreak`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tokenize/punkt.py#L1373)
> `text_contains_sentbreak(self, text)`

Returns True if the given text includes a sentence break.
### `tokenize`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tokenize/punkt.py#L1268)
> `tokenize(self, text, realign_boundaries=True)`

Given a text, returns a list of the sentences in that text.
### `tokenize_sents`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tokenize/api.py#L44)
> `tokenize_sents(self, strings)`

Apply ``self.tokenize()`` to each element of ``strings``.  I.e.:

    return [self.tokenize(s) for s in strings]

:rtype: list(list(str))
### `train`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tokenize/punkt.py#L1252)
> `train(self, train_text, verbose=False)`

Derives parameters from a given training text, or uses the parameters
given. Repeated calls to this method destroy previous parameters. For
incremental training, instantiate a separate PunktTrainer instance.
