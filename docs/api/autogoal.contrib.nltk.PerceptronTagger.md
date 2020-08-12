# `autogoal.contrib.nltk.PerceptronTagger`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/nltk/_generated.py#L137)
> `PerceptronTagger(self)`

Greedy Averaged Perceptron tagger, as implemented by Matthew Honnibal.
See more implementation details here:
    https://explosion.ai/blog/part-of-speech-pos-tagger-in-python

>>> from nltk.tag.perceptron import PerceptronTagger

Train the model

>>> tagger = PerceptronTagger(load=False)

>>> tagger.train([[('today','NN'),('is','VBZ'),('good','JJ'),('day','NN')],
... [('yes','NNS'),('it','PRP'),('beautiful','JJ')]])

>>> tagger.tag(['today','is','a','beautiful','day'])
[('today', 'NN'), ('is', 'PRP'), ('a', 'PRP'), ('beautiful', 'JJ'), ('day', 'NN')]

Use the pretrain model (the default constructor)

>>> pretrain = PerceptronTagger()

>>> pretrain.tag('The quick brown fox jumps over the lazy dog'.split())
[('The', 'DT'), ('quick', 'JJ'), ('brown', 'NN'), ('fox', 'NN'), ('jumps', 'VBZ'), ('over', 'IN'), ('the', 'DT'), ('lazy', 'JJ'), ('dog', 'NN')]

>>> pretrain.tag("The red cat".split())
[('The', 'DT'), ('red', 'JJ'), ('cat', 'NN')]
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `encode_json_obj`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tag/perceptron.py#L256)
> `encode_json_obj(self)`

### `eval`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/sklearn/_builder.py#L50)
> `eval(self)`

### `evaluate`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tag/api.py#L57)
> `evaluate(self, gold)`

Score the accuracy of the tagger against the gold standard.
Strip the tags from the gold standard text, retag it using
the tagger, then compute the accuracy score.

:type gold: list(list(tuple(str, str)))
:param gold: The list of tagged sentences to score the tagger on.
:rtype: float
### `load`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tag/perceptron.py#L247)
> `load(self, loc)`

:param loc: Load a pickled model at location.
:type loc: str
### `normalize`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tag/perceptron.py#L267)
> `normalize(self, word)`

Normalization used in pre-processing.
- All words are lower cased
- Groups of digits of length 4 are represented as !YEAR;
- Other digits are represented as !DIGITS

:rtype: str
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/nltk/_generated.py#L465)
> `run(self, input)`

### `tag`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tag/perceptron.py#L172)
> `tag(self, tokens, return_conf=False, use_tagdict=True)`

Tag tokenized sentences.
:params tokens: list of word
:type tokens: list(str)
### `tag_sents`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tag/api.py#L49)
> `tag_sents(self, sentences)`

Apply ``self.tag()`` to each element of *sentences*.  I.e.:

    return [self.tag(sent) for sent in sentences]
### `train`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tag/perceptron.py#L196)
> `train(self, sentences, save_loc=None, nr_iter=5)`

Train a model from sentences, and save it at ``save_loc``. ``nr_iter``
controls the number of Perceptron training iterations.

:param sentences: A list or iterator of sentences, where each sentence
    is a list of (words, tags) tuples.
:param save_loc: If not ``None``, saves a pickled model in this location.
:param nr_iter: Number of training iterations.
