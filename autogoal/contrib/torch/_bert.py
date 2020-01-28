from transformers import BertModel, BertTokenizer
import torch

from autogoal.kb import Sentence, MatrixContinuousDense
from autogoal.grammar import Discrete
from autogoal.utils import CacheManager


class BertEmbedding:
    """
    Transforms a sentence into a list of vector embeddings using a Bert pretrained English model.

    ##### Notes

    On the first use the model `bert-case-uncased` from [huggingface/transformers](https://github.com/huggingface/transformers)
    will be downloaded. This may take a few minutes.

    If you are using the development container the model should be already downloaded for you.
    """

    def __init__(self, length: Discrete(16, 256)):
        self.model = CacheManager.instance().get(
            "bert-model", lambda: BertModel.from_pretrained("bert-base-uncased")
        )
        self.tokenizer = CacheManager.instance().get(
            "bert-tokenizer", lambda: BertTokenizer.from_pretrained("bert-base-uncased")
        )
        self.length = length

    def run(self, input: Sentence(language="english")) -> MatrixContinuousDense():
        tokens = self.tokenizer.encode(input)
        ids = torch.tensor([tokens])

        with torch.no_grad():
            output = self.model(ids)[0].numpy()

        output = output.reshape((len(tokens), -1))
        return output
