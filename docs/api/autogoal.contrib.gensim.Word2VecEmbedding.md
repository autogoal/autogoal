# `autogoal.contrib.gensim.Word2VecEmbedding`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/gensim/_base.py#L16)
> `Word2VecEmbedding(self)`

This class transform a word in embedding vector using Word2Vec of `gensim` (using `glove-twitter-25`).

##### Notes

On the first use the model Word2Vec of gensim will be downloaded. This may take a few minutes.

If you are using the development container the model should be already downloaded for you.
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/gensim/_base.py#L35)
> `run(self, input)`

This method use Word2Vec of gensim for tranform a word in embedding vector.
        
