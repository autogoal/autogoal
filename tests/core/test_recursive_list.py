import pprint
from autogoal.kb import CategoricalVector, Matrix, MatrixContinuous, MatrixContinuousDense, MatrixContinuousSparse, build_pipeline_graph, build_pipelines, Text, Document, Word, Stem, List, Tuple, Sentence, algorithm
from autogoal.ml.metrics import accuracy
from autogoal.contrib import find_classes
from autogoal.ml import AutoML
from autogoal.grammar import Discrete, generate_cfg, Subset, Categorical
from autogoal.search import PESearch, RandomSearch, RichLogger

class TextAlgorithm:
    def run(
        self, 
        input:Sentence()) -> Document():
            pass

class TestAlgorithm():
    def run(
        self, 
        input:Stem()) -> Tuple(MatrixContinuous(), CategoricalVector()):
            pass

class StemWithDependanceAlgorithm:
    def __init__(
        self, 
        ub:algorithm(Sentence(), Document())
    ):
        pass

    def run(
        self, 
        input:Word()) -> Stem():
            pass

class StemAlgorithm:
    def run(
        self, 
        input:Word()) -> Stem():
        print('inside StemAlgorithm')

class HigherStemAlgorithm:
    def __init__(
        self, 
        stem:algorithm(Word(), Stem())
    ):
        pass

    def run(
        self, 
        input:List(Word())
    ) -> List(Stem()):
        pass


def test_recursive_list_pipeline_graph():
    pipelineBuilder = build_pipelines(input=List(Word())
                                 ,output= List(Stem())
                                 ,registry=[StemAlgorithm, HigherStemAlgorithm])
    search = RandomSearch(
        pipelineBuilder,
        _make_mock_fitness_fn(0,0),
        random_state=0,
        errors="warn"
        )

    best_pipeline_, _ = search.run(3)
    assert best_pipeline_.steps[0].__class__.__name__ == "ListAlgorithm"

    search = RandomSearch(
        pipelineBuilder,
        _make_mock_fitness_fn(0,0),
        random_state=1,
        errors="warn"
        )

    best_pipeline_, _ = search.run(3)
    assert best_pipeline_.steps[0].__class__.__name__ == "HigherStemAlgorithm"

def _make_mock_fitness_fn(X, y):
    def mock_fitness_fn(pipeline):
        return 1
    return mock_fitness_fn
    