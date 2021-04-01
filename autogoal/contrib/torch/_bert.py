from autogoal.kb import AlgorithmBase
from transformers import BertModel, BertTokenizer
from pathlib import Path
import torch
import numpy as np

from autogoal.kb import Sentence, MatrixContinuousDense, Tensor3, Seq, Word
from autogoal.kb import Supervised
from autogoal.grammar import DiscreteValue, CategoricalValue
from autogoal.utils import CacheManager, nice_repr


@nice_repr
class BertEmbedding(AlgorithmBase):
    """
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
    """

    use_cache = False

    def __init__(
        self, merge_mode: CategoricalValue("avg", "first") = "avg", *, verbose=False
    ):  # , length: Discrete(16, 512)):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.verbose = verbose
        self.print("Using device: %s" % self.device)
        self.merge_mode = merge_mode
        self.model = None
        self.tokenizer = None

    def print(self, *args, **kwargs):
        if not self.verbose:
            return

        print(*args, **kwargs)

    def run(self, input: Seq[Word]) -> MatrixContinuousDense:
        if self.model is None:
            self.model = BertModel.from_pretrained("bert-base-multilingual-cased").to(
                self.device
            )
            self.tokenizer = BertTokenizer.from_pretrained(
                "bert-base-multilingual-cased"
            )

        if self.use_cache:
            sentence_hash = hash(" ".join(input))
            cache_path = Path(__file__).parent / f"cached_bert_{sentence_hash}.npy"

            if cache_path.exists():
                return np.load(cache_path)

        bert_tokens = [self.tokenizer.tokenize(x) for x in input]
        bert_senquence = self.tokenizer.encode_plus(
            [t for tokens in bert_tokens for t in tokens], return_tensors="pt"
        )
        with torch.no_grad():
            encoded_tokens, _ = self.model(**bert_senquence)
            encoded_tokens = encoded_tokens.squeeze(0)

        count = 0
        matrix = []
        for i, token in enumerate(input):
            contiguous = len(bert_tokens[i])
            vectors = encoded_tokens[count : count + contiguous, :]
            vector = self._merge(vectors)
            matrix.append(vector)
            count += contiguous

        matrix = np.vstack(matrix)

        if self.use_cache:
            np.save(cache_path, matrix)

        return matrix

    def _merge(self, vectors):
        if not vectors.size(0):
            return np.zeros(vectors.size(1), dtype="float32")
        if self.merge_mode == "avg":
            return vectors.mean(dim=0).numpy()
        elif self.merge_mode == "first":
            return vectors[0, :].numpy()
        else:
            raise ValueError("Unknown merge mode")


@nice_repr
class BertTokenizeEmbedding(AlgorithmBase):
    """
    Transforms a sentence into a list of vector embeddings using a Bert pretrained English model.

    ##### Notes

    On the first use the model `best-base-multilingual-cased` from 
    [huggingface/transformers](https://github.com/huggingface/transformers)
    will be downloaded. This may take a few minutes.

    If you are using the development container the model should be already downloaded for you.
    """

    def __init__(self, verbose=False):  # , length: Discrete(16, 512)):
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

    def run(self, input: Seq[Sentence]) -> Tensor3:
        if self.model is None:
            self.model = BertModel.from_pretrained("bert-base-multilingual-cased").to(
                self.device
            )
            self.tokenizer = BertTokenizer.from_pretrained(
                "bert-base-multilingual-cased"
            )

        self.print("Tokenizing...", end="", flush=True)
        tokens = [
            self.tokenizer.encode(x, max_length=32, pad_to_max_length=True)
            for x in input
        ]
        self.print("done")

        ids = torch.tensor(tokens).to(self.device)

        with torch.no_grad():
            self.print("Embedding...", end="", flush=True)
            output = self.model(ids)[0].numpy()
            self.print("done")

        return output
