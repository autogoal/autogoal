from autogoal.kb import *
from autogoal.kb._algorithm import build_pipeline_graph_old, PipelineSpace
from autogoal.sampling import Sampler
from autogoal.utils import nice_repr
import matplotlib.pyplot as plt
import networkx as nx

import streamlit as st
import networkx as nx
import nx_altair as nxa


class T1:
    pass


class T2:
    pass


class T3:
    pass


class T4:
    pass


class T5:
    pass


class T6:
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
class T2T3_T4(AlgorithmBase):
    def run(self, t2: T2, t3: T3) -> T4:
        pass


@nice_repr
class T4_T5(AlgorithmBase):
    def run(self, t4: T4) -> T5:
        pass


@nice_repr
class T4_T5(AlgorithmBase):
    def run(self, t4: T4) -> T5:
        pass


@nice_repr
class T3T4_T5(AlgorithmBase):
    def run(self, t3: T3, t4: T4) -> T5:
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


input_types = (T1,)
output_type = T5
registry = [T1_T2, T2_T2, T2_T3, T2T3_T4, T3T4_T5]


pipelines_old = build_pipeline_graph_old(
    input_types=input_types, output_type=output_type, registry=registry,
)

pipelines = build_pipeline_graph(
    input_types=input_types, output_type=output_type, registry=registry,
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
    """
)


def sample():
    r = pipelines.sample()
    return r


st.code(sample())

st.button("Sample another pipeline")
