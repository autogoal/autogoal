from autogoal.contrib import find_classes
from autogoal.kb import (
    algorithm,
    Sentence,
    Seq,
    Word,
    Stem,
    build_pipeline_graph,
    Supervised,
    Label,
)
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


from autogoal.contrib.nltk import FeatureSeqExtractor
from autogoal.contrib.sklearn import CRFTagger


def test_crf_pipeline():
    graph = build_pipeline_graph(
        input_types=(Seq[Seq[Word]], Supervised[Seq[Seq[Label]]]),
        output_type=Seq[Seq[Label]],
        registry=[FeatureSeqExtractor, CRFTagger],
    )

    pipeline = graph.sample()
