try:
    import gensim

    # assert gensim.__version__ == "3.8.1"
except:
    print("(!) Code in `autogoal.contrib.gensim` requires `gensim==3.8.1`.")
    print("(!) You can install it with `pip install autogoal[gensim]`.")
    raise


from ._base import (
    Word2VecEmbedding,
    Word2VecEmbeddingSpanish,
    FastTextEmbeddingSpanishSUC,
    FastTextEmbeddingSpanishSWBC,
    GloveEmbeddingSpanishSWBC,
)
