from autogoal.kb import Seq, Word, Sentence, MatrixContinuousDense, build_pipeline_graph
from autogoal.contrib import find_classes
from autogoal.experimental.fasttex._base import UnsupervisedWordRepresentation, UnsupervisedWordRepresentationPT

def test_unsupervised_word_representation_pipeline():
    pipelines = build_pipeline_graph(
        input_types=(Seq[Sentence], Seq[Word]),
        output_type=MatrixContinuousDense,
        registry=find_classes()
        + [UnsupervisedWordRepresentation, UnsupervisedWordRepresentationPT],
    )
    nodes = pipelines.nodes()
    assert UnsupervisedWordRepresentation in nodes

    pipelines = build_pipeline_graph(
        input_types=Seq[Word],
        output_type=MatrixContinuousDense,
        registry=find_classes()
        + [UnsupervisedWordRepresentation, UnsupervisedWordRepresentationPT],
    )
    nodes = pipelines.nodes()
    assert UnsupervisedWordRepresentationPT in nodes


def test_algorithm_correct_types():
    assert UnsupervisedWordRepresentation.input_types() == (
        Seq[Sentence],
        Seq[Word],
    )
    assert UnsupervisedWordRepresentation.output_type() == MatrixContinuousDense

    assert UnsupervisedWordRepresentationPT.input_types() == (Seq[Word],)
    assert UnsupervisedWordRepresentationPT.output_type() == MatrixContinuousDense
