# `autogoal.contrib.gensim.GloveEmbeddingSpanishSWBC`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/contrib/gensim/_base.py#L174)
> `GloveEmbeddingSpanishSWBC(self)`

This class transform a word in embedding vector using Glove of `gensim`.

##### Notes

On the first use the model will be downloaded. This may take a few minutes.
If you are using the development container the model should be already downloaded for you.

##### Examples

>>> embedder = FastTextEmbeddingSpanishSWBC()
>>> embedder.run("algoritmo")
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/contrib/gensim/_base.py#L205)
> `run(self, input)`

This method use FastText of gensim for tranform a word in embedding vector.
        
