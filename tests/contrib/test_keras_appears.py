from autogoal.contrib import find_classes
from autogoal.kb import build_pipeline_graph, MatrixContinuousDense, Supervised, VectorCategorical


def test_matrix_classification_pipeline_can_be_built():
    pipelines = build_pipeline_graph(
        input_types=(MatrixContinuousDense, Supervised[VectorCategorical]),
        output_type=VectorCategorical,
        registry=find_classes("Keras")
    )

    pipeline = pipelines.sample()
