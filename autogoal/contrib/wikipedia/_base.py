import wikipedia

from autogoal.kb import Word, Entity, Summary


class WikipediaSummary:
    """This class find a word in Wikipedia and return a summary in english.
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
    """This class find a word in Wikipedia and return a summary in english.
    """

    def __init__(self):
        pass

    def run(self, input: Word(domain='general', language='english'))-> bool:
        """This method use Word2Vect of gensim for tranform a word in embedding vector.
        """
        return bool(wikipedia.search(input))


class WikipediaSummarySpanish:
    """This class find a word in Wikipedia and return a summary in Spanish.
    """

    def __init__(self):
        wikipedia.set_lang("es")

    def run(self, input: Word(domain='general', language='english'))-> Summary():
        """This method use Word2Vect of gensim for tranform a word in embedding vector.
        """
        try:
            return wikipedia.summary(input)
        except:
            return ""


class WikipediaContainsWordSpanish:
    """This class find a word in Wikipedia and return a summary in Spanish.
    """

    def __init__(self):
        wikipedia.set_lang("es")

    def run(self, input: Word(domain='general', language='english'))-> bool:
        """This method use Word2Vect of gensim for tranform a word in embedding vector.
        """
        return bool(wikipedia.search(input))