from pathlib import Path

import requests
import shutil
import numpy as np

import gensim.downloader as api
from autogoal.kb import ContinuousVector, Word
from autogoal.utils import CacheManager, nice_repr
from autogoal.datasets import download_and_save
from gensim.models import KeyedVectors
from gensim.models.fasttext import FastText, FastTextKeyedVectors


@nice_repr
class Word2VecEmbedding:
    """This class transform a word in embedding vector using Word2Vec of `gensim` (using `glove-twitter-25`).

    ##### Notes

    On the first use the model Word2Vec of gensim will be downloaded. This may take a few minutes.

    If you are using the development container the model should be already downloaded for you.
    """
    def __init__(self):
        self._model = None

    @property
    def model(self) -> KeyedVectors:
        if self._model is None:
            self._model = api.load("glove-twitter-25")

        return self._model

    def run(
        self, input: Word(domain="general", language="english")
    ) -> ContinuousVector():
        """This method use Word2Vec of gensim for tranform a word in embedding vector.
        """
        try:
            return self.model.get_vector(input)
        except:
            return np.zeros(25)


@nice_repr
class Word2VecEmbeddingSpanish:
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

    def run(
        self, input: Word(domain="general", language="spanish")
    ) -> ContinuousVector():
        """This method use Word2Vec of gensim for tranform a word in embedding vector.
        """
        try:
            return self.model.get_vector(input.lower())
        except KeyError:
            return np.zeros(400)


@nice_repr
class FastTextEmbeddingSpanishSUC:
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
        url = "https://zenodo.org/record/3234051/files/embeddings-l-model.bin?download=1"
        path = Path(__file__).parent / "fasttext-spanish-suc.bin"

        download_and_save(url, path)
        return FastText.load_fasttext_format(str(path)).wv

    @property
    def model(self) -> FastTextKeyedVectors:
        if self._model is None:
            self._model = self._load_model()

        return self._model

    def run(
        self, input: Word(domain="general", language="spanish")
    ) -> ContinuousVector():
        """This method use FastText of gensim for tranform a word in embedding vector.
        """
        try:
            return self.model.get_vector(input.lower())
        except KeyError:
            return np.zeros(300)


@nice_repr
class FastTextEmbeddingSpanishSWBC:
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

    def run(
        self, input: Word(domain="general", language="spanish")
    ) -> ContinuousVector():
        """This method use FastText of gensim for tranform a word in embedding vector.
        """
        try:
            return self.model.get_vector(input.lower())
        except KeyError:
            return np.zeros(300)


@nice_repr
class GloveEmbeddingSpanishSWBC:
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

    def run(
        self, input: Word(domain="general", language="spanish")
    ) -> ContinuousVector():
        """This method use FastText of gensim for tranform a word in embedding vector.
        """
        try:
            return self.model.get_vector(input.lower())
        except KeyError:
            return np.zeros(300)
