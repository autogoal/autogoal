import inspect
import warnings
import networkx as nx
from collections import namedtuple

from autogoal.grammar import GraphSpace, Graph, CfgInitializer
from autogoal.utils import nice_repr
from autogoal.kb._data import conforms, build_composite_tuple, Tuple, build_composite_list, List


def build_pipelines(input, output, registry) -> 'PipelineBuilder':
    """
    Creates a `PipelineBuilder` instance that generates all pipelines
    from `input` to `output` types.

    ##### Parameters

    - `input`: type descriptor for the desired input.
    - `output`: type descriptor for the desired output.
    - `registry`: list of available classes to build the pipelines.
    """
    G = Graph()

    open_nodes = []
    closed_nodes = set()

    def connect_tuple_wrappers(node, output_type):
        if not isinstance(output_type, Tuple):
            return

        for index in range(0, len(output_type.inner)):
            internal_input = output_type.inner[index]

            for other_clss in registry:
                annotations = _get_annotations(other_clss)
                other_input = annotations.input

                if not (conforms(internal_input, other_input) and other_clss != node):
                    continue

                # `other_class` has input compatible with one element in the Tuple
                # build the output `Tuple[..., internal_output, ...]` of the wrapper class
                internal_output = annotations.output
                output_tuple = list(output_type.inner)
                output_tuple[index] = internal_output
                output_tuple_type = Tuple(*output_tuple)

                # dynamic class representing the wrapper algorithm
                other_wrapper = build_composite_tuple(index, output_type, output_tuple_type)
                open_nodes.append(other_wrapper)

                G.add_edge(node, other_wrapper)

    list_pairs = set()

    def connect_list_wrappers(node, output_type):
        def connect(internal_output, depth):       
            for other_clss in registry:
                if other_clss == node:
                    continue

                annotations = _get_annotations(other_clss)

                if annotations in list_pairs:
                    continue

                other_input = annotations.input
                other_output = annotations.output

                if other_input == other_output:
                    continue

                if not conforms(internal_output, other_input):
                    continue

                other_wrapper = build_composite_list(other_input, other_output, depth)
                list_pairs.add(annotations)

                print(other_wrapper)

                open_nodes.append(other_wrapper)
                G.add_edge(node, other_wrapper)

        depth = 0

        while isinstance(output_type, List):
            depth += 1
            output_type = output_type.inner
            connect(output_type, depth)

    # Enqueue open nodes
    for clss in registry:
        if conforms(input, _get_annotations(clss).input):
            open_nodes.append(clss)
            G.add_edge(GraphSpace.Start, clss)

    connect_tuple_wrappers(GraphSpace.Start, input)
    connect_list_wrappers(GraphSpace.Start, input)

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
            if conforms(output_type, other_input) and other_clss != clss:
                open_nodes.append(other_clss)
                G.add_edge(clss, other_clss)

        connect_tuple_wrappers(clss, output_type)
        connect_list_wrappers(clss, output_type)

        if conforms(output_type, output):
            G.add_edge(clss, GraphSpace.End)

    if GraphSpace.End not in G:
        raise ValueError("No pipelines can be constructed from input to output.")

    reachable_from_end = set(nx.dfs_preorder_nodes(G.reverse(False), GraphSpace.End))
    unreachable_nodes = set(G.nodes) - reachable_from_end
    G.remove_nodes_from(unreachable_nodes)

    if not GraphSpace.Start in G:
        raise ValueError("No pipelines can be constructed from input to output.")

    return PipelineBuilder(G, registry)


# @nice_repr
class PipelineError(Exception):
    def __init__(self, step, inner):
        super().__init__(step, inner)
        self.step = step
        self.inner = inner


@nice_repr
class Pipeline:
    def __init__(self, steps):
        self.steps = steps

    def send(self, msg: str, *args, **kwargs):
        found = False
        for step in self.steps:
            if hasattr(step, msg):
                getattr(step, msg)(*args, **kwargs)
                found = True
        if not found:
            warnings.warn(f'No step answered message {msg}.')

    def run(self, x):
        for step in self.steps:
            try:
                x = step.run(x)
            except Exception as e:
                raise PipelineError(step=step.__class__.__name__, inner=e)

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
    return conforms(input, _get_annotations(clss).input)


def _has_output(clss, output):
    return conforms(_get_annotations(clss).output, output)
