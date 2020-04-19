import spacy

from autogoal.grammar import Categorical
from autogoal.kb import Sentence, Tuple, Word, Flags, List
from autogoal.utils import nice_repr


@nice_repr
class SpacyNLP:
    def __init__(self, language:Categorical("en", "es")):
        self.language = language

    def run(self, input: Sentence()) -> Tuple(List(Word()), List(Flags())):
        pass
    