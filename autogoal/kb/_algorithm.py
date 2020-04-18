import inspect
import warnings
import networkx as nx
from collections import namedtuple
import logging


from autogoal.grammar import GraphSpace, Graph, CfgInitializer
from autogoal.utils import nice_repr
from autogoal.kb._data import (
    conforms,
    build_composite_tuple,
    Tuple,
    build_composite_list,
    List,
    DataType
)


MAX_LIST_DEPTH = 5


def build_pipeline_graph(input: DataType, output: DataType, registry, max_pipeline_width=5) -> "PipelineBuilder":
    """
    Creates a `PipelineBuilder` instance that generates all pipelines
    from `input` to `output` types.

    ##### Parameters

    - `input`: type descriptor for the desired input.
    - `output`: type descriptor for the desired output.
    - `registry`: list of available classes to build the pipelines.
    """
    
    logger = logging.getLogger("autogoal.build_pipeline_graph")
    
    # First we will unpack the input and output type and
    # store them in actual lists for easier use

    if isinstance(input, Tuple):
        input_type = list(input.inner)
    else:
        input_type = [input]

    if isinstance(output, Tuple):
        output_type = list(output.inner)
    else:
        output_type = [output]

    logger.debug(f"input_type={input_type}")
    logger.debug(f"output_type={output_type}")
    
    # The graph contains all the algorithms, each algorithm is connected
    # to all those nodes that it can process, which are nodes whose output
    # type is a superset of what the algorithm requires.
    G = Graph()

    # For each node stored in the graph, we will store also the full list
    # of all outputs that we can guarantee are available at this point.
    # Initially we add the `Start` node, which produces all of the inputs.
    G.add_node(GraphSpace.Start, types=input_type)

    # We will apply a BFS algorithm at this point. We will make sure
    # that once a node is processed, all the algorithms to which it could
    # potentially connect are stored in the graph.
    # Initially the `Start` node is the only one open.
    open_nodes = [GraphSpace.Start]
    closed_nodes = set()

    while open_nodes:
        # This is the next node we will need to connect.
        node = open_nodes.pop(0)

        if node in closed_nodes:
            continue

        # When leaving this node we can guarantee that we have the types in this list.
        types = G.nodes[node]['types']
        logger.debug(f"Processing node={node.__name__} with types={types}")

        # We will need this method to check if all of the input types of and algorithm are
        # guaranteed at this point, i.e., if they are available in `types`,
        # or at least a conforming type is.
        def type_is_guaranteed(input_type):
            for other_type in types:
                if conforms(other_type, input_type):
                    return True

            return False

        # In this point we have to identify all the algorithms that could continue
        # from this point on. These are all the algorithms whose input expects a subset
        # of the types that we already have.
        for algorithm in registry:
            annotations = _get_annotations(algorithm)
            algorithm_input_types = annotations.input.inner if isinstance(annotations.input, Tuple) else [annotations.input]
            algorithm_output_types = annotations.output.inner if isinstance(annotations.output, Tuple) else [annotations.output]
            logger.debug(f"Analyzing algorithm={algorithm.__name__} with inputs={algorithm_input_types} and outputs={algorithm_output_types}")

            if any(not type_is_guaranteed(input_type) for input_type in algorithm_input_types):
                logger.debug(f"Skipping algorithm={algorithm.__name__}")
                continue
                    
            # At this point we can add the current algorithm to the graph.
            # We have two options for doing this.
            # First, we make the current algorithm "consume" the input types,
            # hence, the output types produced at this point are the output types
            # this algorithm provides plus any input type not consumed so far.
            output_types = [t for t in types if t not in algorithm_input_types] + algorithm_output_types
            logger.debug(f"Adding node={algorithm.__name__} producing types={output_types}")

            # We add this node to the graph and we mark that it consumes the inputs,
            # so that later when sampling we can correctly align all the types
            G.add_node(algorithm, types=output_types, consume=algorithm_input_types)
            G.add_edge(node, algorithm)
            open_nodes.append(algorithm)

        # Let's check if we can add the `End` node.
        if all(type_is_guaranteed(t) for t in output_type):
            G.add_edge(node, GraphSpace.End)
            
        closed_nodes.add(node)

    # Once done we have to check if the `End` node was at some point included in the graph.
    # Otherwise that means there is no possible path.
    if GraphSpace.End not in G:
        raise TypeError(
            "No pipelines can be constructed from input:%r to output:%r."
            % (input, output)
        )

    # Now we remove all nodes that don't participate in any path
    # leaving to `End`
    reachable_from_end = set(nx.dfs_preorder_nodes(G.reverse(False), GraphSpace.End))
    unreachable_nodes = set(G.nodes) - reachable_from_end
    G.remove_nodes_from(unreachable_nodes)

    # If the node `Start` was removed, that means the graph is disconnected.
    if not GraphSpace.Start in G:
        raise TypeError(
            "No pipelines can be constructed from input:%r to output:%r."
            % (input, output)
        )

    return PipelineBuilder(G, registry)
                    

def build_pipelines(input, output, registry) -> "PipelineBuilder":
    """
    Creates a `PipelineBuilder` instance that generates all pipelines
    from `input` to `output` types.

    ##### Parameters

    - `input`: type descriptor for the desired input.
    - `output`: type descriptor for the desired output.
    - `registry`: list of available classes to build the pipelines.
    """

    warnings.warn(
        "This method is deprecated and not under use by AutoGOAL's"
        " internal API anymore, use `build_pipeline_graph` instead.",
        category=DeprecationWarning,
        stacklevel=2,
    )

    list_pairs = set()
    types_queue = []

    if isinstance(input, Tuple):
        types_queue.extend(input.inner)
    else:
        types_queue.append(input)

    if isinstance(output, Tuple):
        types_queue.extend(output.inner)
    else:
        types_queue.append(output)

    types_seen = set()

    while types_queue:
        output_type = types_queue.pop(0)

        def build(internal_output, depth):
            if internal_output in types_seen:
                return

            for other_clss in registry:
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
                registry.append(other_wrapper)
                types_queue.append(_get_annotations(other_wrapper).output)

        depth = 0

        while isinstance(output_type, List):
            if output_type.depth() >= MAX_LIST_DEPTH:
                break

            depth += 1

            output_type = output_type.inner
            build(output_type, depth)
            types_seen.add(output_type)

            print(output_type)

    list_tuples = set()

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
                if (index, output_type, output_tuple_type) in list_tuples:
                    continue

                other_wrapper = build_composite_tuple(
                    index, output_type, output_tuple_type
                )
                list_tuples.add((index, output_type, output_tuple_type))
                registry.append(other_wrapper)

                open_nodes.append(other_wrapper)
                G.add_edge(node, other_wrapper)

    G = Graph()

    open_nodes = []
    closed_nodes = set()

    # Enqueue open nodes
    for clss in registry:
        if conforms(input, _get_annotations(clss).input):
            open_nodes.append(clss)
            G.add_edge(GraphSpace.Start, clss)

    connect_tuple_wrappers(GraphSpace.Start, input)

    if GraphSpace.Start not in G:
        raise TypeError("There are no classes compatible with input type:%r." % input)

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

        if conforms(output_type, output):
            G.add_edge(clss, GraphSpace.End)

    if GraphSpace.End not in G:
        raise TypeError(
            "No pipelines can be constructed from input:%r to output:%r."
            % (input, output)
        )

    reachable_from_end = set(nx.dfs_preorder_nodes(G.reverse(False), GraphSpace.End))
    unreachable_nodes = set(G.nodes) - reachable_from_end
    G.remove_nodes_from(unreachable_nodes)

    if not GraphSpace.Start in G:
        raise TypeError(
            "No pipelines can be constructed from input:%r to output:%r."
            % (input, output)
        )

    import pprint

    pprint.pprint(list(G.nodes))

    # raise ValueError()

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
            elif hasattr(step, "send"):
                step.send(msg, *args, **kwargs)
                found = True

        if not found:
            warnings.warn(f"No step answered message {msg}.")

    def run(self, x):
        for step in self.steps:
            try:
                x = step.run(x)
            except Exception as e:
                raise e from None
                # raise PipelineError(step=step.__class__.__name__, inner=e)

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
