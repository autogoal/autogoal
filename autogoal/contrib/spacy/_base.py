import spacy

from autogoal.grammar import Categorical
from autogoal.kb import Sentence, Tuple, Word, Flags, List, Boolean
from autogoal.utils import nice_repr

from functools import cached_property


@nice_repr
class SpacyNLP:
    def __init__(self, language:Categorical("en", "es"), extract_pos: Boolean()):
        self.language = language

    @cached_property
    def nlp(self):
        return spacy.load(self.language)

    def run(self, input: Sentence()) -> Tuple(List(Word()), List(Flags())):
        tokenized = self.nlp(input)
    
        tokens = []
        flags = []

        for token in tokenized:
            tokens.append(token.text)
            flags.append(dict(
                # TODO: rellenar las cosas que spacy da, post-tag, etc,
            ))

        return tokens, flags
