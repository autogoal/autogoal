# `autogoal.contrib.torch.BertEmbedding`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/contrib/torch/_bert.py#L12)
> `BertEmbedding(self, merge_mode='avg', verbose=False)`

Transforms a sentence already tokenized into a list of vector embeddings using a Bert pretrained multilingual model.

##### Examples

```python
>>> sentence = "embed this wrongword".split()
>>> bert = BertEmbedding(verbose=False)
>>> embedding = bert.run(sentence)
>>> embedding.shape
(3, 768)
>>> embedding
array([[ 0.3887945 , -0.22509816,  0.24768752, ...,  0.7490128 ,
         0.00565467, -0.2144883 ],
       [ 0.1428812 , -0.25218996,  0.19961214, ...,  0.964931  ,
         0.5816741 , -0.2297722 ],
       [ 0.63840234, -0.09097156, -0.80802155, ...,  0.9195696 ,
         0.27364567,  0.14955777]], dtype=float32)

```

##### Notes

On the first use the model `best-base-multilingual-cased` 
from [huggingface/transformers](https://github.com/huggingface/transformers)
will be downloaded. This may take a few minutes.

If you are using the development container the model should be already downloaded for you.
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/utils/__init__.py#L87)
> `repr_method(self)`

### `print`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/contrib/torch/_bert.py#L54)
> `print(self, *args, **kwargs)`

### `run`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/contrib/torch/_bert.py#L60)
> `run(self, input)`

