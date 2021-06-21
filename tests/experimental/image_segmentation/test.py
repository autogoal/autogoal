from autogoal.ml import AutoML
from autogoal.kb import *
from autogoal.experimental.image_segmentation.segmentation._semantics import ImageFile, ImageMask, Image
from autogoal.contrib import find_classes
from autogoal.experimental.image_segmentation.segmentation._base import ImageSegmenter, ImagePreprocessor
from autogoal.experimental.image_segmentation.keras._base import KerasImageSegmenter
from autogoal.experimental.image_segmentation.dataset.dataset import load
from autogoal.search import RichLogger
from autogoal.utils import Min, Gb


def test_uses_created_clases():
    pipelines = build_pipeline_graph(
        input_types=(Seq[ImageFile], Supervised[Seq[ImageMask]]),
        output_type=Seq[ImageMask],
        registry=find_classes("Keras") + [ImageSegmenter, ImagePreprocessor, KerasImageSegmenter]
    )
    nodes = pipelines.nodes()
    assert ImageSegmenter in nodes

    pipelines = build_pipeline_graph(
        input_types=ImageFile,
        output_type=Image,
        registry=find_classes("Keras") + [ImageSegmenter, ImagePreprocessor, KerasImageSegmenter]
    )
    nodes = pipelines.nodes()
    assert ImagePreprocessor in nodes


def test_algotithm_correct_types():
    assert ImageSegmenter.input_types() == (
        Seq[ImageFile],
        Supervised[Seq[ImageMask]],
    )
    assert ImageSegmenter.output_type() == Seq[ImageMask]

    assert ImagePreprocessor.input_types() == (ImageFile,)
    assert ImagePreprocessor.output_type() == Image


def test():
    automl = AutoML(
        input=(Seq[ImageFile], Supervised[Seq[ImageMask]]),
        output=Seq[ImageMask],
        registry=find_classes() + [ImageSegmenter, ImagePreprocessor, KerasImageSegmenter],
        evaluation_timeout=10 * Min,
        memory_limit=3.5 * Gb,
        search_timeout=30 * Min,
    )

    x_train, y_train, x_test, y_test = load()
    automl.fit(x_train, y_train, logger=[RichLogger()])
    score = automl.score(x_test, y_test)
    print(score)
