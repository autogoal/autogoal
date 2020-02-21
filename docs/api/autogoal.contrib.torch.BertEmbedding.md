# `autogoal.contrib.torch.BertEmbedding`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/contrib/torch/_bert.py#L11)
> `BertEmbedding(self, verbose=False)`

Transforms a sentence already tokenized into a list of vector embeddings using a Bert pretrained English model.

##### Examples

```python
>>> sentence = "the show must go on".split()
>>> bert = BertEmbedding(verbose=False)
>>> embedding = bert.run(sentence)
Creating cached object 'bert-model'
Creating cached object 'bert-tokenizer'
>>> embedding.shape
(5, 768)
>>> embedding
array([[-0.36865586, -0.09041885, -0.05140949, ...,  0.1486538 ,
         0.5336794 ,  0.336316  ],
       [-0.09966173, -0.05827313,  0.30103225, ..., -0.14690986,
         0.0892544 , -0.12143768],
       [-0.04454202,  0.4275659 ,  0.34425724, ..., -0.07058787,
         0.05012058,  0.18611997],
       [-0.10367895, -0.14797121,  0.29116577, ..., -0.14221254,
        -0.29068246,  0.16387418],
       [ 0.03009991, -0.17941667,  0.37870008, ..., -0.01924773,
        -0.12460218,  0.16398118]], dtype=float32)

```

##### Notes

On the first use the model `best-base-multilingual-cased` from [huggingface/transformers](https://github.com/huggingface/transformers)
will be downloaded. This may take a few minutes.

If you are using the development container the model should be already downloaded for you.
### `repr_method`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/utils/__init__.py#L91)
> `repr_method(self)`

### `print`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/contrib/torch/_bert.py#L56)
> `print(self, *args, **kwargs)`

### `run`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/contrib/torch/_bert.py#L62)
> `run(self, input)`

