from autogoal.kb import (
    Document,
    Sentence,
    Seq,
    Stem,
    Word,
    build_pipeline_graph,
    algorithm,
    AlgorithmBase,
)
from autogoal.search import RandomSearch


class TextAlgorithm(AlgorithmBase):
    def run(self, input: Sentence) -> Document:
        pass


class StemWithDependanceAlgorithm(AlgorithmBase):
    def __init__(self, ub: algorithm(Sentence, Document)):
        pass

    def run(self, input: Word) -> Stem:
        pass


class StemAlgorithm(AlgorithmBase):
    def run(self, input: Word) -> Stem:
        print("inside StemAlgorithm")


class HigherStemAlgorithm(AlgorithmBase):
    def __init__(self, stem: algorithm(Word, Stem)):
        pass

    def run(self, input: Seq[Word]) -> Seq[Stem]:
        pass


def test_recursive_list_pipeline_graph():
    pipelineBuilder = build_pipeline_graph(
        input_types=Seq[Word],
        output_type=Seq[Stem],
        registry=[StemAlgorithm, HigherStemAlgorithm],
    )


def _make_mock_fitness_fn(X, y):
    def mock_fitness_fn(pipeline):
        return 1

    return mock_fitness_fn
