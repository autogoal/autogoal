import inspect
import networkx as nx
from collections import namedtuple

from autogoal.grammar import GraphSpace, Graph, CfgInitializer
from autogoal.utils import nice_repr


def build_pipelines(input, output, registry):
    G = Graph()

    open_nodes = []
    closed_nodes = set()

    # Enqueue open nodes
    for clss in registry:
        if input.conforms(_get_annotations(clss).input):
            open_nodes.append(clss)
            G.add_edge(GraphSpace.Start, clss)

    if GraphSpace.Start not in G:
        raise ValueError("There are no classes compatible with input type.")

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

    if GraphSpace.End not in G:
        raise ValueError("No pipelines can be constructed from input to output.")

    reachable_from_end = set(nx.dfs_preorder_nodes(G.reverse(False), GraphSpace.End))
    unreachable_nodes = set(G.nodes) - reachable_from_end
    G.remove_nodes_from(unreachable_nodes)

    if not GraphSpace.Start in G:
        raise ValueError("No pipelines can be constructed from input to output.")

    return PipelineBuilder(G, registry)


@nice_repr
class Pipeline:
    def __init__(self, steps):
        self.steps = steps

    def send(self, msg: str, *args, **kwargs):
        for step in self.steps:
            if hasattr(step, msg):
                getattr(step, msg)(*args, **kwargs)

    def run(self, x):
        for step in self.steps:
            x = step.run(x)

        return x


class PipelineBuilder(GraphSpace):
    def __init__(self, graph, registry):
        super().__init__(graph, initializer=CfgInitializer(registry=registry))

    def sample(self, *args, **kwargs) -> Pipeline:
        path = super().sample(*args, **kwargs)
        return Pipeline(path)



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
