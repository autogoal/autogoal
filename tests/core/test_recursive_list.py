from autogoal.kb import Document, Sentence, Seq, Stem, Word, build_pipeline_graph, algorithm
from autogoal.search import RandomSearch


class TextAlgorithm:
    def run(self, input: Sentence) -> Document:
        pass


class StemWithDependanceAlgorithm:
    def __init__(self, ub: algorithm(Sentence, Document)):
        pass

    def run(self, input: Word) -> Stem:
        pass


class StemAlgorithm:
    def run(self, input: Word) -> Stem:
        print("inside StemAlgorithm")


class HigherStemAlgorithm:
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
    search = RandomSearch(
        pipelineBuilder, _make_mock_fitness_fn(0, 0), random_state=0, errors="warn"
    )

    best_pipeline_, _ = search.run(3)
    assert best_pipeline_.steps[0].__class__.__name__ == "ListAlgorithm"

    search = RandomSearch(
        pipelineBuilder, _make_mock_fitness_fn(0, 0), random_state=1, errors="warn"
    )

    best_pipeline_, _ = search.run(3)
    assert best_pipeline_.steps[0].__class__.__name__ == "HigherStemAlgorithm"


def _make_mock_fitness_fn(X, y):
    def mock_fitness_fn(pipeline):
        return 1

    return mock_fitness_fn
