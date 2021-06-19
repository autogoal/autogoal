from autogoal.kb import build_pipeline_graph, AlgorithmBase
from autogoal.experimental.gan.gan import GAN
from autogoal.contrib import find_classes
from autogoal.experimental.gan.utils import ImageFile
from autogoal.kb import Seq, Word, Supervised


def test_algorithm_in_pipeline_graph():
    pipelines = build_pipeline_graph(
        input_types=([Seq[ImageFile], Seq[ImageFile]]),
        output_type=ImageFile,
        registry=find_classes() + [GAN],
    )

    assert GAN in pipelines.nodes()


def test_algorithm_report_correct_types():
    assert GAN.input_types() == (Seq[ImageFile], Seq[ImageFile],)

    assert GAN.output_type() == ImageFile
