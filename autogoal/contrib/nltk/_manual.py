from autogoal.grammar import Continuous, Discrete, Categorical, Boolean
from autogoal.contrib.sklearn._builder import SklearnWrapper, SklearnTransformer
from autogoal.kb import *
from numpy import inf, nan

from gensim.models.doc2vec import Doc2Vec as _Doc2Vec
class Doc2Vec(_Doc2Vec, SklearnTransformer):
    def __init__(
        self,
        dm: Discrete(min=0, max=2),
        dbow_words: Discrete(min=-100, max=100),
        dm_concat: Discrete(min=-100, max=100),
        dm_tag_count: Discrete(min=0, max=2),
        alpha:Continuous(min=0.001, max = 0.075),
        epochs:Discrete(min=2, max=10),
        window:Discrete(min=2, max=10)
    ):
        
        # self.dm=dm
        # self.dbow_words=dbow_words
        # self.dm_concat=dm_concat
        # self.dm_tag_count=dm_tag_count

        super().__init__(
            dm=dm,
            dbow_words=dbow_words,
            dm_concat=dm_concat,
            dm_tag_count=dm_tag_count,
            alpha=alpha,
            epochs=epochs,
            window=window
        )

    def fit_transform(
        self, 
        X, 
        y=None
    ):
        self.fit(X, y=None)
        return self.transform(X)

    def fit(
        self, 
        X,
        y
    ):
        #Data must be turned to tagged data as TaggedDocument(List(Token), Tag)
        #Tag use to be an unique integer

        from gensim.models.doc2vec import TaggedDocument as _TaggedDocument
        tagged_data = [_TaggedDocument(X[i], str(i)) for i in range(len(X))]

        self.build_vocab(tagged_data)
        return self.train(tagged_data, total_examples=self.corpus_count, epochs=self.epochs)


    def transform(
        self,
        X,
        y=None  
    ):
        return [self.infer_vector(x) for x in X]

    def run(self, input: List(Document())) -> MatrixContinuousDense():
       """This methods receive a document list and transform this into a dense continuous matrix. 
       """
       return SklearnTransformer.run(self, input)
    
from nltk.corpus import stopwords
class StopwordRemover(SklearnWrapper):
    def __init__(
        self,
        language:Categorical('danish',\
                             'dutch',\
                             'english',\
                             'finnish',\
                             'french',\
                             'german',\
                             'hungarian',\
                             'italian',\
                             'norwegian',\
                             'portuguese',\
                             'russian',\
                             'spanish',\
                             'swedish',\
                             'turkish')
    ):
        self.language = language
        self.words = stopwords.words(language)
        
    def fit(
        self, 
        X, 
        y=None
    ):
        pass
    
    def fit_transform(
        self, 
        X, 
        y=None
    ):
        self.fit(X, y=None)
        return self.transform(X)

    def transform(
        self,
        X,
        y=None  
    ):
        #Considering data as list of tokenized documents
        return [word for word in X if word not in self.words]
    
    def _train(self, input):
        X, y = input
        return [word for word in X if word not in self.words], y

    def _eval(self, input):
        X, y = input
        return [word for word in X if word not in self.words], y
    
    def run(self, input: List(Word())) -> List(Word()):
       """This methods receive a word list list and transform this into a word list list without stopwords. 
       """
       return SklearnTransformer.run(self, input)
        
# class TextLowerer(SklearnTransformer):
#     def __init__(
#         self
#     ):
#         pass
    
#     def fit(
#         self, 
#         X, 
#         y=None
#     ):
#         pass
    
#     def fit_transform(
#         self, 
#         X, 
#         y=None
#     ):
#         self.fit(X, y=None)
#         return self.transform(X)

#     def transform(
#         self,
#         X,
#         y=None  
#     ):
#         #Considering data as list of raw documents
#         return [str.lower(x) for x in X]
    
#     def run(self, input: Word()) -> Word():
#        """This methods receive a document list and transform this into a document list with lowered case. 
#        """
#        return SklearnTransformer.run(self, input)