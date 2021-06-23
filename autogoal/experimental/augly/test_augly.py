from autogoal.kb import (
    build_pipeline_graph,
    Seq,
    Sentence,
    Supervised,
    Categorical,
)
from autogoal.contrib import find_classes
from autogoal.experimental.augly import (
    SimulateTypos,
    Grayscale,
    AugLyImage # semantic type for image
)

def test_auglytransformer_pipeline():
    pipelines_text = build_pipeline_graph(
        input_types=(Seq[Sentence], Supervised[Categorical]),
        output_type=Seq[Sentence],
        registry=[SimulateTypos] + find_classes()
    )

    tnodes = pipelines_text.nodes()

    assert SimulateTypos in tnodes

    pipelines_image = build_pipeline_graph(
        input_types=(AugLyImage, Supervised[Categorical]),
        output_type=AugLyImage,
        registry=[Grayscale] + find_classes()
    )

    inodes = pipelines_image.nodes()

    assert Grayscale in inodes

def test_auglytransformer_types():
    assert SimulateTypos.input_types()[0] == Seq[Sentence] # only check the X set
    assert SimulateTypos.output_type() == Seq[Sentence]

    assert Grayscale.input_types()[0] == AugLyImage
    assert Grayscale.output_type() == AugLyImage