import pytest

import numpy as np
from autogoal.contrib import find_classes
from autogoal.ml import *
from autogoal.experimental.pipeline import *
from autogoal.experimental.semantics import *


class T1:
    pass


class T2:
    pass


class T3:
    pass


class T4:
    pass


@nice_repr
class T1_T2(AlgorithmBase):
    def run(self, t1: T1) -> T2:
        pass


@nice_repr
class T2_T3(AlgorithmBase):
    def run(self, t2: T2) -> T3:
        pass


@nice_repr
class T3_T4(AlgorithmBase):
    def run(self, t3: T3) -> T4:
        pass


@nice_repr
class T2_T3_Supervised(AlgorithmBase):
    def run(self, x: T2, y: Supervised[T3]) -> T3:
        pass


def test_build_pipeline_with_two_algorithms():
    pipeline_builder = build_pipeline_graph(
        input_types=(T1,), output_type=T3, registry=[T1_T2, T2_T3]
    )
    pipeline = pipeline_builder.sample()

    assert repr(pipeline.algorithms) == "[T1_T2(), T2_T3()]"


def test_build_pipeline_with_supervised():
    pipeline_builder = build_pipeline_graph(
        input_types=(T1, Supervised[T3],),
        output_type=T3,
        registry=[T1_T2, T2_T3_Supervised],
    )
    pipeline = pipeline_builder.sample()

    assert repr(pipeline.algorithms) == "[T1_T2(), T2_T3_Supervised()]"


def test_build_pipeline_has_no_extra_nodes():
    pipeline_builder = build_pipeline_graph(
        input_types=(T1,), output_type=T3, registry=[T1_T2, T2_T3, T3_T4]
    )

    print(pipeline_builder.graph.nodes)

    assert "T3_T4" not in [
        node.algorithm.__name__
        for node in pipeline_builder.graph
        if hasattr(node, "algorithm")
    ]


class Float2Str(AlgorithmBase):
    def run(self, a: float) -> str:
        return str(a)


class StrToInt(AlgorithmBase):
    def run(self, b: str) -> int:
        return len(b)


def test_when_pipeline_has_two_algorithms_then_passes_the_output():
    pipeline = Pipeline([Float2Str(), StrToInt()], input_types=[float])
    result = pipeline.run(3.0)
    assert result == 3


class TwoInputAlgorithm(AlgorithmBase):
    def run(self, a: int, b: str) -> int:
        return a * len(b)


def test_when_pipeline_step_has_more_that_one_input_then_all_arguments_are_passed():
    pipeline = Pipeline([TwoInputAlgorithm()], input_types=[int, str])
    assert pipeline.run(3, "hello world") == 33


def test_when_pipeline_second_step_receives_two_input_one_from_previous_and_one_from_origin():
    pipeline = Pipeline([StrToInt(), TwoInputAlgorithm()], input_types=[str])
    result = pipeline.run("hello world")
    assert result == 121


@pytest.mark.slow
def test_build_real_pipeline():
    graph = build_pipeline_graph(
        input_types=(MatrixContinuous, Supervised[VectorCategorical]),
        output_type=VectorCategorical,
        registry=find_classes(),
    )
    pipeline = graph.sample(sampler=Sampler(random_state=42))
    pipeline.run(np.ones(shape=(2, 2)), [0, 1])


class A(AlgorithmBase):
    def run(self, x: MatrixContinuous):
        pass


def test_build_input_args_with_subclass():
    m = np.ones(shape=(2, 2))

    result = build_input_args(A, {MatrixContinuousDense: m})
    assert id(result["x"]) == id(m)


@pytest.mark.slow
def test_automl_finds_classifiers():
    automl = AutoML(
        input=(MatrixContinuous, Supervised[VectorCategorical]),
        output=VectorCategorical,
    )
    builder = automl.make_pipeline_builder()

    assert len(builder.graph) > 10


@pytest.mark.slow
def test_automl_trains_pipeline():
    automl = AutoML(
        input=(MatrixContinuous, Supervised[VectorCategorical]),
        output=VectorCategorical,
        search_iterations=1,
        random_state=42,
    )
    automl.fit(np.ones(shape=(2, 2)), [0, 1])
    automl.predict(np.ones(shape=(2, 2)))

