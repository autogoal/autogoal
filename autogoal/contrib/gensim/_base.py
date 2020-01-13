import gensim.downloader as api

model = api.load("glove-twitter-25")

class Word2VectEmbedding: 
    """This class transform a word in embedding vector using Word2Vect of gensim (using glove-twitter-25).
    """

    def __init__(self):
        self.model = api.load("glove-twitter-25")

    def convert(self, word):
        """This method use Word2Vect of gensim for tranform a word in embedding vector. 
        """
        return self.model.get_vector(word)

    def run(self, input:"Word")-> "Vector": 
        return convert(input)


