# `autogoal.contrib.nltk.TnT`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/nltk/_generated.py#L264)
> `TnT(self, Trained, N, C)`

TnT - Statistical POS tagger

IMPORTANT NOTES:

* DOES NOT AUTOMATICALLY DEAL WITH UNSEEN WORDS

  - It is possible to provide an untrained POS tagger to
    create tags for unknown words, see __init__ function

* SHOULD BE USED WITH SENTENCE-DELIMITED INPUT

  - Due to the nature of this tagger, it works best when
    trained over sentence delimited input.
  - However it still produces good results if the training
    data and testing data are separated on all punctuation eg: [,.?!]
  - Input for training is expected to be a list of sentences
    where each sentence is a list of (word, tag) tuples
  - Input for tag function is a single sentence
    Input for tagdata function is a list of sentences
    Output is of a similar form

* Function provided to process text that is unsegmented

  - Please see basic_sent_chop()


TnT uses a second order Markov model to produce tags for
a sequence of input, specifically:

  argmax [Proj(P(t_i|t_i-1,t_i-2)P(w_i|t_i))] P(t_T+1 | t_T)

IE: the maximum projection of a set of probabilities

The set of possible tags for a given word is derived
from the training data. It is the set of all tags
that exact word has been assigned.

To speed up and get more precision, we can use log addition
to instead multiplication, specifically:

  argmax [Sigma(log(P(t_i|t_i-1,t_i-2))+log(P(w_i|t_i)))] +
         log(P(t_T+1|t_T))

The probability of a tag for a given word is the linear
interpolation of 3 markov models; a zero-order, first-order,
and a second order model.

  P(t_i| t_i-1, t_i-2) = l1*P(t_i) + l2*P(t_i| t_i-1) +
                         l3*P(t_i| t_i-1, t_i-2)

A beam search is used to limit the memory usage of the algorithm.
The degree of the beam can be changed using N in the initialization.
N represents the maximum number of possible solutions to maintain
while tagging.

It is possible to differentiate the tags which are assigned to
capitalized words. However this does not result in a significant
gain in the accuracy of the results.
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

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
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/nltk/_generated.py#L561)
> `run(self, input)`

### `tag`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tag/tnt.py#L285)
> `tag(self, data)`

Tags a single sentence

:param data: list of words
:type data: [string,]

:return: [(word, tag),]

Calls recursive function '_tagword'
to produce a list of tags

Associates the sequence of returned tags
with the correct words in the input sequence

returns a list of (word, tag) tuples
### `tag_sents`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tag/api.py#L49)
> `tag_sents(self, sentences)`

Apply ``self.tag()`` to each element of *sentences*.  I.e.:

    return [self.tag(sent) for sent in sentences]
### `tagdata`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tag/tnt.py#L267)
> `tagdata(self, data)`

Tags each sentence in a list of sentences

:param data:list of list of words
:type data: [[string,],]
:return: list of list of (word, tag) tuples

Invokes tag(sent) function for each sentence
compiles the results into a list of tagged sentences
each tagged sentence is a list of (word, tag) tuples
### `train`

> [📝](/usr/local/lib/python3.6/dist-packages/nltk/tag/tnt.py#L134)
> `train(self, data)`

Uses a set of tagged data to train the tagger.
If an unknown word tagger is specified,
it is trained on the same data.

:param data: List of lists of (word, tag) tuples
:type data: tuple(str)
