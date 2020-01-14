import gensim.downloader as api


class Word2VectEmbedding: 
    """This class transform a word in embedding vector using Word2Vect of `gensim` (using `glove-twitter-25`).

    ##### Notes

    On the first use the model Word2Vect of gensim will be downloaded. This may take a few minutes.
    
    If you are using the development container the model should be already downloaded for you.
    """

    def __init__(self):
        self.model = api.load("glove-twitter-25")

    def run(self, input:Word(domain='general', language='english'))-> "Vector": 
        """This method use Word2Vect of gensim for tranform a word in embedding vector. 
        """
        return self.model.get_vector(input)
