from pathlib import Path

import requests

import gensim.downloader as api
from autogoal.kb import ContinuousVector, Word
from autogoal.utils import CacheManager, nice_repr
from gensim.models import KeyedVectors


@nice_repr
class Word2VecEmbedding:
    """This class transform a word in embedding vector using Word2Vect of `gensim` (using `glove-twitter-25`).

    ##### Notes

    On the first use the model Word2Vect of gensim will be downloaded. This may take a few minutes.

    If you are using the development container the model should be already downloaded for you.
    """

    @property
    def model(self) -> KeyedVectors:
        return CacheManager.instance().get(
            "glove-twitter-25", lambda: api.load("glove-twitter-25")
        )

    def run(
        self, input: Word(domain="general", language="english")
    ) -> ContinuousVector():
        """This method use Word2Vect of gensim for tranform a word in embedding vector.
        """
        return self.model.get_vector(input)


@nice_repr
class Word2VecEmbeddingSpanish:
    """This class transform a word in embedding vector using Word2Vect of `gensim` (using `Spanish 3B Word2Vec`).

    ##### Notes

    On the first use the model Word2Vect of gensim will be downloaded. This may take a few minutes.

    If you are using the development container the model should be already downloaded for you.
    """

    def _load_model(self):
        url = "https://zenodo.org/record/1410403/files/keyed_vectors.zip?download=1"
        path = Path(__file__).parent / "spanish-w2v.kv"

        if not path.exists():
            stream = requests.get(url)

            with path.open("wb") as fp:
                fp.write(stream.content)

        return KeyedVectors.load(str(path), mmap="r")

    @property
    def model(self) -> KeyedVectors:
        return CacheManager.instance().get("spanish-w2v", self._load_model)

    def run(
        self, input: Word(domain="general", language="spanish")
    ) -> ContinuousVector():
        """This method use Word2Vect of gensim for tranform a word in embedding vector.
        """
        return self.model.get_vector(input)
