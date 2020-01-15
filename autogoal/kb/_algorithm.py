import inspect
import networkx as nx
from collections import namedtuple

from autogoal.grammar import GraphSpace, Graph, CfgInitializer


def build_pipelines(input, output, registry):
    G = Graph()

    # Initialize the graph
    G.add_node(GraphSpace.Start)
    G.add_node(GraphSpace.End)

    open_nodes = []
    closed_nodes = set()

    # Enqueue open nodes
    for clss in registry:
        if input.conforms(_get_annotations(clss).input):
            open_nodes.append(clss)
            G.add_edge(GraphSpace.Start, clss)

    while open_nodes:
        clss = open_nodes.pop(0)

        if clss in closed_nodes:
            continue

        closed_nodes.add(clss)
        output_type = _get_annotations(clss).output

        for other_clss in registry:
            other_input = _get_annotations(other_clss).input
            if output_type.conforms(other_input) and other_clss != clss:
                open_nodes.append(other_clss)
                G.add_edge(clss, other_clss)

        if output_type.conforms(output):
            G.add_edge(clss, GraphSpace.End)

    reachable_from_end = set(nx.dfs_preorder_nodes(G.reverse(False), GraphSpace.End))
    unreachable_nodes = set(G.nodes) - reachable_from_end
    G.remove_nodes_from(unreachable_nodes)

    return PipelineSpace(G, registry)


class PipelineSpace(GraphSpace):
    def __init__(self, graph, registry):
        super().__init__(graph, initializer=CfgInitializer(registry=registry))

    def sample(self, *args, **kwargs):
        path = super().sample(*args, **kwargs)
        return path


Annotations = namedtuple("Annotations", ["input", "output"])


def _get_annotations(clss):
    run_method = clss.run
    input_type = inspect.signature(run_method).parameters["input"].annotation
    output_type = inspect.signature(run_method).return_annotation

    return Annotations(input=input_type, output=output_type)


def _has_input(clss, input):
    return input.conforms(_get_annotations(clss).input)


def _has_output(clss, output):
    return _get_annotations(clss).output.conforms(output)
