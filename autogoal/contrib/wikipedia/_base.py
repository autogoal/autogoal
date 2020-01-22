import wikipedia

from autogoal.kb import Word, Entity, Summary


class WikipediaSummary:
    """This class find a word in Wikipedia and return a summary.
    """

    def __init__(self):
        pass

    def run(self, input: Word(domain='general', language='english'))-> Summary():
        """This method use Word2Vect of gensim for tranform a word in embedding vector.
        """
        try:
            return wikipedia.summary(input)
        except:
            return ""


class WikipediaContainsWord:
    """This class find a word in Wikipedia and return a summary.
    """

    def __init__(self):
        pass

    def run(self, input: Entity(domain='general', language='english'))-> bool:
        """This method use Word2Vect of gensim for tranform a word in embedding vector.
        """
        return bool(wikipedia.search(input))