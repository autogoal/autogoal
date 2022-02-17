from autogoal.ml import AutoML
from autogoal.kb import *
from autogoal.kb._algorithm import build_pipeline_graph_old, PipelineSpace
from autogoal.utils import nice_repr
import matplotlib.pyplot as plt
import networkx as nx
from autogoal.contrib import find_classes
from autogoal.contrib.keras._generated import (
    Activation,
    Conv1D,
    Conv2D,
    Dense,
    BatchNormalization,
    Dropout,
    Embedding,
    Flatten,
    MaxPooling2D,
    Reshape2D,
    Seq2SeqLSTM,
    Seq2VecBiLSTM,
    Seq2VecLSTM,
    Seq2SeqBiLSTM,
    TimeDistributed,
)
from autogoal.contrib.nltk import FeatureSeqExtractor
from autogoal.contrib.sklearn import CRFTagger

import sys
import streamlit as st
import textwrap
import networkx as nx
import pandas as pd
import altair as alt
import nx_altair as nxa
import inspect


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
class T2_T2(AlgorithmBase):
    def run(self, x: T2) -> T2:
        pass


@nice_repr
class T2_T2_V2(AlgorithmBase):
    def run(self, x: T2) -> T2:
        pass


@nice_repr
class T2_T3_Supervised(AlgorithmBase):
    def run(self, x: T2, y: Supervised[T3]) -> T3:
        pass


# pipelines = build_pipeline_graph(
#     input_types=(MatrixContinuousDense,),
#     output_type=MatrixContinuousDense,
#     registry=[ExactAlgorithm, HigherInputAlgorithm, LowerOutputAlgorithm],
# )

# pipelines = build_pipeline_graph(
#     input_types=(MatrixContinuousDense, Supervised[VectorCategorical]),
#     output_type=VectorCategorical,
#     registry=find_classes("Keras"),
# )

# pipelines = build_pipeline_graph(
#     input_types=(Seq[Seq[Word]], Supervised[Seq[Seq[Label]]]),
#     output_type=Seq[Seq[Label]],
#     registry=[FeatureSeqExtractor, CRFTagger],
# )

# pipelines = build_pipeline_graph(
#     input_types=(Seq[Text],),
#     output_type=Document,
#     registry=[
#         WordToWordAlgorithm,
#         TextToWordAlgorithm,
#         WordToWordListAlgorithm,
#         WordListToSentenceAlgorithm,
#         WordListToSentenceAlgorithm,
#         SentenceListToDocumentAlgorithm,
#         TextListToDocumentAlgorithm,
#     ],
# )

# pipelines = build_pipeline_graph_v2(
#     input_types=(MatrixContinuousDense,),
#     output_type=MatrixContinuousDense,
#     registry=[ExactAlgorithm, HigherInputAlgorithm, LowerOutputAlgorithm],
# )


pipelines_old = build_pipeline_graph_old(
    input_types=(T1, Supervised[T3],),
    output_type=T3,
    registry=[T1_T2, T2_T2, T2_T2_V2, T2_T3_Supervised],
)

pipelines = build_pipeline_graph(
    input_types=(T1, Supervised[T3],),
    output_type=T3,
    registry=[T1_T2, T2_T2, T2_T2_V2, T2_T3_Supervised],
)

# graph = pipelines.graph
# nx.draw(graph, with_labels=True)
# plt.show()


def get_node_repr(node):
    try:
        return get_node_repr(node.inner)
    except:
        return dict(
            label=str(node).split(".")[-1], module=node.__module__.split("_")[0]
        )


def plot_graph(space: PipelineSpace):
    graph = nx.DiGraph()

    for node in space.graph.nodes:
        attrs = get_node_repr(node)
        graph.add_node(attrs["label"], **attrs)

    for u, v in space.graph.edges:
        graph.add_edge(get_node_repr(u)["label"], get_node_repr(v)["label"])

    pos = nx.nx_pydot.pydot_layout(graph, prog="dot", root=space.Start)
    chart = (
        nxa.draw_networkx(graph, pos=pos, node_color="module", node_tooltip="label")
        .properties(height=500)
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)


st.write(
    """
    ### Old build_pipeline_graph
    """
)

plot_graph(pipelines_old)

st.write(
    """
    ### New build_pipeline_graph
    """
)

plot_graph(pipelines)

st.write(
    """
    ### Example Pipeline
    
    Here is an example pipeline that has been randomly sampled from the previous graph.
    You can try different samples. Notice how not only the nodes (algorithms) that participate
    in the pipeline are different each time, but also their internal hyperparameters change.
    
    When sampling a pipeline from the graph AutoGOAL samples all the internal
    hyperparameters as defined by the constructor.
    When these hyperparameters have complex values (e.g., an algorithm per-se), AutoGOAL
    recursively samples instances of the internal algorithms, and so on.
    """
)

st.code(pipelines.sample())

st.button("Sample another pipeline")
