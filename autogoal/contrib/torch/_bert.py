from transformers import BertModel, BertTokenizer
import torch
import numpy as np

from autogoal.kb import Sentence, MatrixContinuousDense, Tensor3, List, Word
from autogoal.grammar import Discrete
from autogoal.utils import CacheManager, nice_repr


@nice_repr
class BertEmbedding:
    """
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
    """

    def __init__(self, verbose=True):  # , length: Discrete(16, 512)):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.verbose = verbose
        self.print("Using device: %s" % self.device)
        self.model = None
        self.tokenizer = None

    def print(self, *args, **kwargs):
        if not self.verbose:
            return

        print(*args, **kwargs)

    def run(self, input: List(Word(language="english"))) -> MatrixContinuousDense():
        if self.model is None:
            self.model = CacheManager.instance().get(
                "bert-model",
                lambda: BertModel.from_pretrained("bert-base-multilingual-cased").to(self.device),
            )
            self.tokenizer = CacheManager.instance().get(
                "bert-tokenizer",
                lambda: BertTokenizer.from_pretrained("bert-base-multilingual-cased"),
            )

        self.print("Tokenizing...", end="", flush=True)
        tokens = self.tokenizer.convert_tokens_to_ids(input)
        self.print("done")

        ids = torch.tensor(tokens).unsqueeze(0).to(self.device)

        with torch.no_grad():
            self.print("Embedding...", end="", flush=True)
            output = self.model(ids)[0].numpy().reshape(len(tokens), -1)
            self.print("done")

        return output


@nice_repr
class BertTokenizeEmbedding:
    """
    Transforms a sentence into a list of vector embeddings using a Bert pretrained English model.

    ##### Notes

    On the first use the model `best-base-multilingual-cased` from [huggingface/transformers](https://github.com/huggingface/transformers)
    will be downloaded. This may take a few minutes.

    If you are using the development container the model should be already downloaded for you.
    """

    def __init__(self):  # , length: Discrete(16, 512)):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        print("Using device: %s" % self.device)
        self.model = None
        self.tokenizer = None

    def run(self, input: List(Sentence(language="english"))) -> Tensor3():
        if self.model is None:
            self.model = CacheManager.instance().get(
                "bert-model",
                lambda: BertModel.from_pretrained("bert-base-multilingual-cased").to(self.device),
            )
            self.tokenizer = CacheManager.instance().get(
                "bert-tokenizer",
                lambda: BertTokenizer.from_pretrained("bert-base-multilingual-cased"),
            )

        print("Tokenizing...", end="", flush=True)
        tokens = [
            self.tokenizer.encode(x, max_length=32, pad_to_max_length=True)
            for x in input
        ]
        print("done")

        ids = torch.tensor(tokens).to(self.device)

        with torch.no_grad():
            print("Embedding...", end="", flush=True)
            output = self.model(ids)[0].numpy()
            print("done")

        return output
