import wikipedia

from autogoal.kb import Word, Document, FeatureSet
from autogoal.utils import nice_repr
from autogoal.kb import AlgorithmBase


@nice_repr
class WikipediaSummary(AlgorithmBase):
    """This class find a word in Wikipedia and return a summary in english.
    """

    def __init__(self):
        pass

    def run(self, input: Word) -> Document:
        """This method use Word2Vect of gensim for tranform a word in embedding vector.
        """
        try:
            return wikipedia.summary(input)
        except:
            return ""


@nice_repr
class WikipediaContainsWord(AlgorithmBase):
    """This class find a word in Wikipedia and return a summary in english.
    """

    def __init__(self):
        pass

    def run(self, input: Word) -> FeatureSet:
        """This method use Word2Vect of gensim for tranform a word in embedding vector.
        """
        return dict(in_wikipedia=bool(wikipedia.search(input)))


@nice_repr
class WikipediaSummarySpanish(AlgorithmBase):
    """This class find a word in Wikipedia and return a summary in Spanish.
    """

    def __init__(self):
        wikipedia.set_lang("es")

    def run(self, input: Word) -> Document:
        """This method use Word2Vect of gensim for tranform a word in embedding vector.
        """
        try:
            return wikipedia.summary(input)
        except:
            return ""


@nice_repr
class WikipediaContainsWordSpanish(AlgorithmBase):
    """This class find a word in Wikipedia and return a summary in Spanish.
    """

    def __init__(self):
        wikipedia.set_lang("es")

    def run(self, input: Word) -> FeatureSet:
        """This method use Word2Vect of gensim for tranform a word in embedding vector.
        """
        return dict(in_wikipedia=bool(wikipedia.search(input)))
