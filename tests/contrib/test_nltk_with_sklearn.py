from autogoal.contrib import find_classes
from autogoal.kb import algorithm, Sentence, Seq, Word, Stem
from autogoal.grammar import generate_cfg, Symbol


class Algorithm:
    def __init__(
        self,
        tokenizer: algorithm(Sentence, Seq[Word]),
        stemmer: algorithm(Word, Stem),
        stopword: algorithm(Seq[Word], Seq[Word]),
    ) -> None:
        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.stopword = stopword


def test_find_nltk_implementations():
    grammar = generate_cfg(Algorithm, find_classes(include=["*.nltk.*"]))

    assert Symbol("Algorithm[[Sentence],Seq[Word]]") in grammar._productions
    assert Symbol("Algorithm[[Word],Stem]") in grammar._productions
    assert Symbol("Algorithm[[Seq[Word]],Seq[Word]]") in grammar._productions
