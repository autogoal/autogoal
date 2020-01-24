from autogoal.grammar import Continuous, Discrete, Categorical, Boolean, Synset
from autogoal.kb._data import *
from numpy import inf, nan

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet


from gensim.models.doc2vec import Doc2Vec as _Doc2Vec
class Doc2Vec(_Doc2Vec):
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
        y=None
    ):
        pass    

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

    #def run(self, input: Document(domain='general')) -> List(Sentence()):
    #    """This methods recive a document and transform this in a list of sentences. 
    #    """
    #    return self.tokenize(input)
    
from nltk.corpus import stopwords
class StopwordRemover():
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
        return [[word for word in document if word not in self.words] for document in X]
        
class TextLowerer():
    def __init__(
        self
    ):
        pass
    
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
        #Considering data as list of raw documents
        return [str.lower(x) for x in X]


class WordnetConcept():
    """Find a word in Wordnet and return a List of Synset de Wordnet
    """

    def __init__(self):
        pass

    def run(self, input: Word(domain='general', language='english'))-> List(Synset()):
        """Find a word in Wordnet and return a List of Synset de Wordnet
        """
        synsets = wordnet.synsets(input)
        names_synsets = []
        for i in synsets:
            names_synsets.append(i.name())

        return names_synsets


class ConverSynset2Word():
    """Recive a Synset of nltk and return de Lemma of this
    """
    
    def __init__(self):
        pass

    def run(self, input: Synset(domain='general', language='english'))-> Word():
        """Recive a Synset of nltk and return de Lemma of this
        """
        return Lemma(input)

from nltk.corpus import sentiwordnet as swn

class SentimentWord():
    """Find a word in SentiWordnet and return a List of sentiment of the word.
    """

    def __init__(self):
        pass

    def run(self, input: Synset(domain='general', language='english'))-> Sentiment():
        """Find a word in SentiWordnet and return a List of sentiment of the word.
        """
        swn_synset = swn.senti_synset(input)
        
        sentiment = {}
        sentiments["positive"] = i.pos_score()
        sentiments["negative"] = i.neg_score()

        return sentiment
