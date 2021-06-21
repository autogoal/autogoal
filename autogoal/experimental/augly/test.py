from autogoal.kb import (
    build_pipeline_graph,
    Seq,
    Sentence,
    Supervised,
    Categorical,
)
from autogoal.contrib import find_classes
from autogoal.experimental.augly import SimulateTypos

def test_auglytransformer_pipeline():
    pipelines = build_pipeline_graph(
        input_types=(Seq[Sentence], Supervised[Categorical]),
        output_type=Seq[Sentence],
        registry=[SimulateTypos] + find_classes()
    )

    pnodes = pipelines.nodes()

    assert SimulateTypos in pnodes

def test_auglytransformer_types():
    assert SimulateTypos.input_types()[0] == Seq[Sentence] # only check the X set
    assert SimulateTypos.output_type() == Seq[Sentence]