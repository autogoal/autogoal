from autogoal.kb import AlgorithmBase
from pathlib import Path

import requests
import shutil
import numpy as np

import gensim.downloader as api
from autogoal.kb import Word, VectorContinuous
from autogoal.utils import CacheManager, nice_repr
from autogoal.datasets import download_and_save
from autogoal.grammar import CategoricalValue, DiscreteValue
from gensim.models import KeyedVectors
from gensim.models.fasttext import FastText, FastTextKeyedVectors


@nice_repr
class Word2VecSmallEmbedding(AlgorithmBase):
    """This class transforms a word in embedding vector using Word2Vec of `gensim` with "smallish" vetor sizes, varying from 25 to 100.

    ##### Notes

    On the first use the model Word2Vec of gensim will be downloaded. This may take a few minutes.

    If you are using the development container the model should be already downloaded for you.
    """

    def __init__(
        self,
        model_name: CategoricalValue(
            "glove-wiki-gigaword-50",
            "glove-wiki-gigaword-100",
            "glove-twitter-25",
            "glove-twitter-50",
            "glove-twitter-100",
        ),
    ):
        self.model_name = model_name
        self._model = None
        sizes = {
            "fasttext-wiki-news-subwords-300": 300,
            "conceptnet-numberbatch-17-06-300": 300,
            "word2vec-google-news-300": 300,
            "glove-wiki-gigaword-50": 50,
            "glove-wiki-gigaword-100": 100,
            "glove-wiki-gigaword-200": 200,
            "glove-wiki-gigaword-300": 300,
            "glove-twitter-25": 25,
            "glove-twitter-50": 50,
            "glove-twitter-100": 100,
            "glove-twitter-200": 200,
        }
        self._size = sizes[model_name]

    @property
    def model(self) -> KeyedVectors:
        if self._model is None:
            self._model = api.load(self.model_name)

        return self._model

    def run(self, input: Word) -> VectorContinuous:
        """This method uses Word2Vec of gensim for tranform a word in embedding vector.
        """
        try:
            return self.model.get_vector(input)
        except:
            return np.zeros(self._size)


@nice_repr
class Word2VecEmbedding(AlgorithmBase):
    """This class transform a word in embedding vector using Word2Vec of `gensim`.

    ##### Notes

    On the first use the model Word2Vec of gensim will be downloaded. This may take a few minutes.

    If you are using the development container the model should be already downloaded for you.
    """

    def __init__(
        self,
        model_name: CategoricalValue(
            "fasttext-wiki-news-subwords-300",
            "conceptnet-numberbatch-17-06-300",
            "word2vec-ruscorpora-300",
            "word2vec-google-news-300",
            "glove-wiki-gigaword-50",
            "glove-wiki-gigaword-100",
            "glove-wiki-gigaword-200",
            "glove-wiki-gigaword-300",
            "glove-twitter-25",
            "glove-twitter-50",
            "glove-twitter-100",
            "glove-twitter-200",
        ),
    ):
        self.model_name = model_name
        self._model = None
        sizes = {
            "fasttext-wiki-news-subwords-300": 300,
            "conceptnet-numberbatch-17-06-300": 300,
            "word2vec-ruscorpora-300": 300,
            "word2vec-google-news-300": 300,
            "glove-wiki-gigaword-50": 50,
            "glove-wiki-gigaword-100": 100,
            "glove-wiki-gigaword-200": 200,
            "glove-wiki-gigaword-300": 300,
            "glove-twitter-25": 25,
            "glove-twitter-50": 50,
            "glove-twitter-100": 100,
            "glove-twitter-200": 200,
        }
        self._size = sizes[model_name]

    @property
    def model(self) -> KeyedVectors:
        if self._model is None:
            self._model = api.load(self.model_name)

        return self._model

    def run(self, input: Word) -> VectorContinuous:
        """This method use Word2Vec of gensim for tranform a word in embedding vector.
        """
        try:
            return self.model.get_vector(input)
        except:
            return np.zeros(self._size)


@nice_repr
class Word2VecEmbeddingSpanish(AlgorithmBase):
    """This class transform a word in embedding vector using Word2Vec of `gensim` (using `Spanish 3B Word2Vec`).

    ##### Notes

    On the first use the model Word2Vec of gensim will be downloaded. This may take a few minutes.

    If you are using the development container the model should be already downloaded for you.
    """

    def __init__(self):
        self._model = None

    def _load_model(self):
        url = "https://zenodo.org/record/1410403/files/keyed_vectors.zip?download=1"
        path = Path(__file__).parent / "spanish-w2v.zip"
        kv = Path(__file__).parent / "complete.kv"

        if download_and_save(url, path):
            shutil.unpack_archive(str(path), str(path.parent))

        return KeyedVectors.load(str(kv), mmap="r")

    @property
    def model(self) -> KeyedVectors:
        if self._model is None:
            self._model = self._load_model()

        return self._model

    def run(self, input: Word) -> VectorContinuous:
        """This method use Word2Vec of gensim for tranform a word in embedding vector.
        """
        try:
            return self.model.get_vector(input.lower())
        except KeyError:
            return np.zeros(400)


@nice_repr
class FastTextEmbeddingSpanishSUC(AlgorithmBase):
    """This class transform a word in embedding vector using FastText of `gensim`.

    ##### Notes

    On the first use the model will be downloaded. This may take a few minutes.
    If you are using the development container the model should be already downloaded for you.

    ##### Examples

    >>> embedder = FastTextEmbeddingSpanishSUC()
    >>> embedder.run("algoritmo")

    """

    def __init__(self):
        self._model = None

    def _load_model(self):
        url = (
            "https://zenodo.org/record/3234051/files/embeddings-l-model.bin?download=1"
        )
        path = Path(__file__).parent / "fasttext-spanish-suc.bin"

        download_and_save(url, path)
        return FastText.load_fasttext_format(str(path)).wv

    @property
    def model(self) -> FastTextKeyedVectors:
        if self._model is None:
            self._model = self._load_model()

        return self._model

    def run(self, input: Word) -> VectorContinuous:
        """This method use FastText of gensim for tranform a word in embedding vector.
        """
        try:
            return self.model.get_vector(input.lower())
        except KeyError:
            return np.zeros(300)


@nice_repr
class FastTextEmbeddingSpanishSWBC(AlgorithmBase):
    """This class transform a word in embedding vector using FastText of `gensim`.

    ##### Notes

    On the first use the model will be downloaded. This may take a few minutes.
    If you are using the development container the model should be already downloaded for you.

    ##### Examples

    >>> embedder = FastTextEmbeddingSpanishSWBC()
    >>> embedder.run("algoritmo")

    """

    def __init__(self):
        self._model = None

    def _load_model(self):
        url = "http://dcc.uchile.cl/~jperez/word-embeddings/fasttext-sbwc.bin"
        path = Path(__file__).parent / "fasttext-spanish-swbc.bin"

        download_and_save(url, path)
        return FastText.load_fasttext_format(str(path)).wv

    @property
    def model(self) -> FastTextKeyedVectors:
        if self._model is None:
            self._model = self._load_model()

        return self._model

    def run(self, input: Word) -> VectorContinuous:
        """This method use FastText of gensim for tranform a word in embedding vector.
        """
        try:
            return self.model.get_vector(input.lower())
        except KeyError:
            return np.zeros(300)


@nice_repr
class GloveEmbeddingSpanishSWBC(AlgorithmBase):
    """This class transform a word in embedding vector using Glove of `gensim`.

    ##### Notes

    On the first use the model will be downloaded. This may take a few minutes.
    If you are using the development container the model should be already downloaded for you.

    ##### Examples

    >>> embedder = FastTextEmbeddingSpanishSWBC()
    >>> embedder.run("algoritmo")

    """

    def __init__(self):
        self._model = None

    def _load_model(self):
        url = "http://dcc.uchile.cl/~jperez/word-embeddings/glove-sbwc.i25.bin"
        path = Path(__file__).parent / "glove-spanish-swbc.bin"

        download_and_save(url, path)
        return FastText.load_fasttext_format(str(path)).wv

    @property
    def model(self) -> FastTextKeyedVectors:
        if self._model is None:
            self._model = self._load_model()

        return self._model

    def run(self, input: Word) -> VectorContinuous:
        """This method use FastText of gensim for tranform a word in embedding vector.
        """
        try:
            return self.model.get_vector(input.lower())
        except KeyError:
            return np.zeros(300)
