from autogoal.contrib.gensim._base import Word2VecEmbeddingSpanish, Word2VecEmbedding

embedding = Word2VecEmbeddingSpanish()
print(len(embedding.run('perro')))

embedding = Word2VecEmbedding()
print(len(embedding.run('dog')))

