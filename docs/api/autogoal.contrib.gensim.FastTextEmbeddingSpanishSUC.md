# `autogoal.contrib.gensim.FastTextEmbeddingSpanishSUC`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/gensim/_base.py#L88)
> `FastTextEmbeddingSpanishSUC(self)`

This class transform a word in embedding vector using FastText of `gensim`.

##### Notes

On the first use the model will be downloaded. This may take a few minutes.
If you are using the development container the model should be already downloaded for you.

##### Examples

>>> embedder = FastTextEmbeddingSpanishSUC()
>>> embedder.run("algoritmo")
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/gensim/_base.py#L119)
> `run(self, input)`

This method use FastText of gensim for tranform a word in embedding vector.
        
