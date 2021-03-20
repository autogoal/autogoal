import pytest

from autogoal.contrib import find_classes
from autogoal.kb import *
from autogoal.experimental.pipeline import Supervised
from autogoal.sampling import Sampler


def _build_pipeline():
    builder = build_pipeline_graph(
        input_types=(Seq[Seq[FeatureSet]], Supervised[Seq[Seq[Label]]]),
        output_type=Seq[Seq[Label]],
        registry=find_classes(include="CRF")
    )

    return builder.sample(sampler=Sampler(random_state=0))


@pytest.mark.contrib
def test_crf_is_found():
    pipeline = _build_pipeline()
    assert "CRFTagger" in repr(pipeline)


@pytest.mark.contrib
def test_crf_training():
    pipeline = _build_pipeline()

    X = [[{"A":1, "B":2, "C":3}] * 5]
    y = [["T", "F", "T", "F", "T"]]

    pipeline.send("train")
    pipeline.run(X, y)

    pipeline.send("eval")
    ypred = pipeline.run(X, None)

    assert ypred == y
