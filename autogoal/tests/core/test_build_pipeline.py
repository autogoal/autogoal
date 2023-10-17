import pytest
from autogoal.ml import AutoML
from autogoal.kb import *


class ExactAlgorithm(AlgorithmBase):
    def run(self, input: MatrixContinuousDense) -> MatrixContinuousDense:
        pass


class HigherInputAlgorithm(AlgorithmBase):
    def run(self, input: MatrixContinuous) -> MatrixContinuousDense:
        pass


class LowerOutputAlgorithm(AlgorithmBase):
    def run(self, input: MatrixContinuousDense) -> MatrixContinuous:
        pass


class WordToWordAlgorithm(AlgorithmBase):
    def run(self, input: Word) -> Word:
        pass


class TextToWordAlgorithm(AlgorithmBase):
    def run(self, input: Text) -> Word:
        pass


class WordToWordListAlgorithm(AlgorithmBase):
    def run(self, input: Word) -> Seq[Word]:
        pass


class WordListToSentenceAlgorithm(AlgorithmBase):
    def run(self, input: Seq[Word]) -> Sentence:
        pass


class SentenceListToDocumentAlgorithm(AlgorithmBase):
    def run(self, input: Seq[SemanticType]) -> Document:
        pass


class TextListToDocumentAlgorithm(AlgorithmBase):
    def run(self, input: Seq[Text]) -> Document:
        pass


# NOTE: This test is coupled to a previous implementation detail
# def assert_graph(graph, start_out, end_in, nodes_amount):
#     """
#     Assert amount of nodes, adjacents of PipelineStart node
#     and in-edges of PipelineEnd node"""
#     start_node = list(graph)[1] #PipelineStart node have fixed position
#     end_node = [node for node in list(graph) if node.__class__.__name__ == "PipelineEnd"][0] #PipelineEnd node

#     assert(graph.out_degree(start_node) == start_out)
#     assert(graph.in_degree(end_node) == end_in)
#     assert(graph.number_of_nodes() == nodes_amount)


def assert_pipeline_graph_failed(input, output, registry):
    pipeline_builder = build_pipeline_graph(
        input_types=input, output_type=output, registry=registry
    )


def test_meta_pipeline_seq():
    # Test List algorithm generation
    build_pipeline_graph(
        input_types=(Seq[Word],), output_type=Seq[Word], registry=[WordToWordAlgorithm]
    )


def test_meta_pipeline_tuple():
    # Test Tuple breakdown feature
    build_pipeline_graph(
        input_types=(Word, Matrix), output_type=Text, registry=[WordToWordAlgorithm]
    )


def test_meta_pipeline_seq_and_tuple():
    # Test Tuple breakdown feature and List algorithm generation
    build_pipeline_graph(
        input_types=(Seq[Word], Matrix),
        output_type=Seq[Word],
        registry=[WordToWordAlgorithm],
    )


def test_simple_pipeline_graph():
    graph = build_pipeline_graph(
        input_types=(MatrixContinuousDense,),
        output_type=MatrixContinuousDense,
        registry=[ExactAlgorithm, HigherInputAlgorithm, LowerOutputAlgorithm],
    ).graph
    # assert_graph(graph, 3, 3, 6)

    graph = build_pipeline_graph(
        input_types=(Seq[Text],),
        output_type=Document,
        registry=[
            WordToWordAlgorithm,
            TextToWordAlgorithm,
            WordToWordListAlgorithm,
            WordListToSentenceAlgorithm,
            WordListToSentenceAlgorithm,
            SentenceListToDocumentAlgorithm,
            TextListToDocumentAlgorithm,
        ],
    ).graph
    # assert_graph(graph, 2, 2, 12)

    graph = build_pipeline_graph(
        input_types=(Seq[Word],),
        output_type=Document,
        registry=[
            WordToWordAlgorithm,
            TextToWordAlgorithm,
            WordToWordListAlgorithm,
            WordListToSentenceAlgorithm,
            WordListToSentenceAlgorithm,
            SentenceListToDocumentAlgorithm,
            TextListToDocumentAlgorithm,
        ],
    ).graph
    # assert_graph(graph, 2, 1, 10)
