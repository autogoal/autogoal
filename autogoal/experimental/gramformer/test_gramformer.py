from autogoal.kb import build_pipeline_graph, Seq, Sentence, Categorical, Supervised
from autogoal.experimental.gramformer import GramCorrect
from autogoal.contrib import find_classes

def test_gramformer_pipeline():
    pl = build_pipeline_graph(
        input_types=(Seq[Sentence], Supervised[Categorical]),
        output_type=Seq[Sentence],
        registry=[GramCorrect] + find_classes()
    )

    nodes = pl.nodes()
    assert GramCorrect in nodes

def test_gramformer_types():
    assert GramCorrect.input_types()[0] == Seq[Sentence]
    assert GramCorrect.output_types() == Seq[Sentence]