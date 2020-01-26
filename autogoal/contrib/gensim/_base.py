import gensim.downloader as api

from gensim.models import KeyedVectors
from autogoal.kb import Word, ContinuousVector
from autogoal.utils import CacheManager, nice_repr


@nice_repr
class Word2VecEmbedding:
    """This class transform a word in embedding vector using Word2Vect of `gensim` (using `glove-twitter-25`).

    ##### Notes

    On the first use the model Word2Vect of gensim will be downloaded. This may take a few minutes.

    If you are using the development container the model should be already downloaded for you.
    """
    @property
    def model(self) -> KeyedVectors:
        return CacheManager.instance().get('glove-twitter-25', lambda: api.load("glove-twitter-25"))

    def run(self, input: Word(domain='general', language='english'))-> ContinuousVector():
        """This method use Word2Vect of gensim for tranform a word in embedding vector.
        """
        return self.model.get_vector(input)
