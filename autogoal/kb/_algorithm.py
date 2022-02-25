import abc
import inspect
import types
import warnings
from collections import OrderedDict, defaultdict, namedtuple
from typing import Any, Collection, Dict, List, Optional, Sequence, Set, Tuple, Type

import networkx as nx
from autogoal.grammar import Graph, GraphSpace, generate_cfg
from autogoal.grammar._graph import End as GraphSpaceEnd
from autogoal.kb._semantics import SemanticType, Seq
from autogoal.sampling import ISampler, Sampler
from autogoal.utils import nice_repr

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol


class Supervised(SemanticType):
    """Represents a supervised version of some type X.

    It is considered a subclass of X for semantic purposes, but not the other way around:

    # >>> issubclass(Supervised[Vector], Vector)
    # True
    # >>> issubclass(Vector, Supervised[Vector])
    # False
    # >>> issubclass(Supervised[Seq[Vector]], Seq[Vector])
    # True
    # >>> issubclass(Seq[Vector], Supervised[Seq[Vector]])
    # False

    """

    __internal_types = {}

    @classmethod
    def _specialize(cls, internal_type):
        try:
            return cls.__internal_types[internal_type]
        except KeyError:
            pass

        class SupervisedImp(Supervised):
            __internal = internal_type

            @classmethod
            def _name(cls):
                return f"Supervised[{cls.__internal}]"

            @classmethod
            def _reduce(cls):
                return Supervised._specialize, (internal_type,)

        cls.__internal_types[internal_type] = SupervisedImp

        return SupervisedImp


def algorithm(*annotations):
    from autogoal.grammar import Symbol, Union

    *inputs, output = annotations

    def match(cls):
        if not hasattr(cls, "run"):
            return False

        signature = inspect.signature(cls.run)
        input_types = [
            v.annotation for k, v in signature.parameters.items() if k != "self"
        ]
        output_type = signature.return_annotation

        if len(inputs) != len(input_types):
            return False

        for expected, real in zip(inputs, input_types):
            if not issubclass(expected, real):
                return False

        if not issubclass(output_type, output):
            return False

        return True

    @classmethod
    def is_compatible(cls, other):
        return match(other)

    @classmethod
    def generate_cfg(cls, grammar, head):
        symbol = head or Symbol(cls.__name__)
        compatible = []

        for _, other_cls in grammar.namespace.items():
            if cls.is_compatible(other_cls):
                compatible.append(other_cls)

        if not compatible:
            raise ValueError(
                f"Cannot find any suitable implementation of algorithms with inputs: {inputs} and output: {output}"
            )

        return Union(symbol.name, *compatible).generate_cfg(grammar, symbol)

    def build(ns):
        ns["generate_cfg"] = generate_cfg
        ns["is_compatible"] = is_compatible

    return types.new_class(f"Algorithm[{inputs},{output}]", bases=(), exec_body=build)


class Algorithm(Protocol):
    """Represents an abstract algorithm with a run method.

    Provides introspection for the expected semantic input and output types.
    Users should inherit from `AlgorithmBase` instead of this class.
    """

    @classmethod
    @abc.abstractmethod
    def input_types(cls) -> Tuple[type]:
        """Returns an ordered list of the expected semantic input types of the `run` method.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def input_args(cls) -> Tuple[str]:
        """Returns an ordered tuple of the names of the arguments in the `run` method.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def output_type(cls) -> type:
        """Returns an ordered list of the expected semantic output type of the `run` method.
        """
        pass

    @abc.abstractmethod
    def run(self, *args):
        """Executes the algorithm.
        """
        pass

    @classmethod
    def is_compatible_with(cls: "Algorithm", input_types):
        """
        Determines if the current algorithm is compatible with a set of input types,
        i.e., if among those types we can find all the necessary inputs for this algorithm.

        >>> class A(AlgorithmBase):
        ...     def run(self, x:int) -> float:
        ...         pass
        >>> A.is_compatible_with([int])
        True
        """
        my_inputs = cls.input_types()

        for needed in my_inputs:
            for having in input_types:
                if issubclass(having, needed):
                    break
            else:
                return False

        return True


class AlgorithmBase(Algorithm, abc.ABC):
    """Represents an algorithm,

    Automatically implements the input and output introspection methods using the `inspect` module.
    Users inheriting from this class must provide type annotations in the `run` method.
    """

    @classmethod
    def input_types(cls) -> Tuple[type]:
        # if not hasattr(cls, "__run_signature__"):
        #     cls.__run_signature__ = inspect.signature(cls.run)

        return tuple(
            param.annotation
            for name, param in inspect.signature(cls.run).parameters.items()
            if name != "self"
        )

    @classmethod
    def input_args(cls) -> Tuple[str]:
        # if not hasattr(cls, "__run_signature__"):
        #     cls.__run_signature__ = inspect.signature(cls.run)

        return tuple(
            name for name in inspect.signature(cls.run).parameters if name != "self"
        )

    @classmethod
    def output_type(cls) -> type:
        # if not hasattr(cls, "__run_signature__"):
        #     cls.__run_signature__ = inspect.signature(cls.run)

        return inspect.signature(cls.run).return_annotation


def make_seq_algorithm(algorithm: Algorithm) -> Algorithm:
    """Lift an algorithm with input types T1, T2, Tn to a meta-algorithm with types Seq[T1], Seq[T2], ...

    The generated class correctly defines the input and output types.
    These implementations are compatible with `build_input_args`:

    >>> class A(AlgorithmBase):
    ...     def __init__(self, alpha):
    ...         self.alpha = 0.5
    ...     def run(self, x:int, y:str) -> float:
    ...         return self.alpha * (x + len(y))
    ...     def __repr__(self):
    ...         return f"A({self.alpha})"
    >>> B = make_seq_algorithm(A)
    >>> b = B(0.5)
    >>> b
    SeqAlgorithm[A(0.5)]
    >>> b.run([1, 2], y=["A", "BC"])
    [1.0, 2.0]
    >>> B.input_types()
    (Seq[<class 'int'>], Seq[<class 'str'>])
    >>> B.input_args()
    ('x', 'y')
    >>> b.output_type()
    Seq[<class 'float'>]
    >>> build_input_args(B, {Seq[int]: [1, 2], Seq[str]: ["hello", "world"]})
    {'x': [1, 2], 'y': ['hello', 'world']}
    >>> b.get_inner_signature()
    <Signature (self, alpha)>
    """

    output_type = algorithm.output_type()

    name = f"SeqAlgorithm[{algorithm.__name__}]"

    def init_method(self, *args, **kwargs):
        self.inner = algorithm(*args, **kwargs)

    def run_method(self, *args, **kwargs) -> Seq[output_type]:
        args_kwargs = _make_list_args_and_kwargs(*args, **kwargs)
        return [self.inner.run(*t.args, **t.kwargs) for t in args_kwargs]

    def repr_method(self):
        return f"SeqAlgorithm[{repr(self.inner)}]"

    def getattr_method(self, attr):
        return getattr(self.inner, attr)

    @classmethod
    def get_inner_signature_method(cls):
        if getattr(algorithm, "get_inner_signature", None):
            return algorithm.get_inner_signature()
        return inspect.signature(algorithm.__init__)

    @classmethod
    def input_types_method(cls):
        return tuple(Seq[t] for t in algorithm.input_types())

    @classmethod
    def input_args_method(cls):
        return algorithm.input_args()

    @classmethod
    def output_types_method(cls):
        return Seq[algorithm.output_type()]

    def body(ns):
        ns["__init__"] = init_method
        ns["run"] = run_method
        ns["__repr__"] = repr_method
        ns["__getattr__"] = getattr_method
        ns["input_types"] = input_types_method
        ns["input_args"] = input_args_method
        ns["output_type"] = output_types_method
        ns["get_inner_signature"] = get_inner_signature_method

    return types.new_class(name=name, bases=(Algorithm,), exec_body=body)


Akw = namedtuple("Akw", ["args", "kwargs"])


def _make_list_args_and_kwargs(*args, **kwargs):
    """Transforms a list of args into individual args and kwargs for an internal algorithm. 

    To be used by `make_seq_algorithm"

    >>> _make_list_args_and_kwargs([1,2], [4,5])
    [Akw(args=(1, 4), kwargs={}), Akw(args=(2, 5), kwargs={})]
    >>> _make_list_args_and_kwargs(x=[1,2], y=[4,5])
    [Akw(args=(), kwargs={'x': 1, 'y': 4}), Akw(args=(), kwargs={'x': 2, 'y': 5})]
    >>> _make_list_args_and_kwargs([1,2], y=[4,5])
    [Akw(args=(1,), kwargs={'y': 4}), Akw(args=(2,), kwargs={'y': 5})]

    """
    lengths = set(len(v) for v in kwargs.values()) | set(len(v) for v in args)

    if len(lengths) != 1:
        raise ValueError("All args and kwargs must be sequences of the same length.")

    length = lengths.pop()

    inner_args = []

    for i in range(length):
        inner_args.append(tuple([xs[i] for xs in args]))

    inner_kwargs = []

    for i in range(length):
        inner_kwargs.append({k: v[i] for k, v in kwargs.items()})

    return [Akw(xs, ks) for xs, ks in zip(inner_args, inner_kwargs)]


def build_input_args(algorithm: Algorithm, values: Dict[type, Any]):
    """Buils the correct input mapping for `algorithm` using the provided `values` mapping types to objects.

    The input can be a class that inherits from `Algorithm` or an instance of such a class.

    >>> class A(AlgorithmBase):
    ...    def run(self, a:int, b:str):
    ...        pass
    >>> values = { str:"hello", float:3.0, int:42 }
    >>> build_input_args(A, values)
    {'a': 42, 'b': 'hello'}
    >>> build_input_args(A(), values)
    {'a': 42, 'b': 'hello'}

    """

    result = {}

    for name, type in zip(algorithm.input_args(), algorithm.input_types()):
        try:
            result[name] = values[type]
        except KeyError:
            for key in values.keys():
                if issubclass(key, type):
                    result[name] = values[key]
                    break
            else:
                raise TypeError(f"Cannot find compatible input value for {type}")

    return result


class PipelineNode:
    __slots__ = ("algorithm", "input_types", "output_types", "grammar")

    def __init__(
        self,
        algorithm: Algorithm,
        input_types,
        output_types,
        registry=None,
        has_grammar: bool = True,
    ) -> None:
        self.algorithm = algorithm
        self.input_types = set(input_types)
        self.output_types = set(output_types)
        self.grammar = (
            generate_cfg(self.algorithm, registry=registry) if has_grammar else None
        )

    def sample(self, sampler: ISampler):
        return self.grammar.sample(sampler=sampler)

    @property
    def __name__(self):
        return self.algorithm.__name__

    def __eq__(self, o: object) -> bool:
        return isinstance(o, PipelineNode) and all(
            [o.algorithm == self.algorithm, o.input_types == self.input_types,]
        )

    def __repr__(self) -> str:
        algorithm_name = _get_name(self.algorithm)

        return f"<PipelineNode(algorithm={algorithm_name},input_types={[i.__name__ for i in self.input_types]},output_types={[o.__name__ for o in self.output_types]})>"

    def __hash__(self) -> int:
        return hash(repr(self))


class NodesTypes(Collection):
    """Auxiliar class that keep track used PipelineNodes and all PipelineNode that have specific output type"""

    __slots__ = ("_nodes", "_types_registry")

    def __init__(self, input_types: Optional[List[type]] = None) -> None:
        self._nodes: Set[PipelineNode] = set()
        self._types_registry: Dict[type, Dict[PipelineNode, None]] = defaultdict(
            OrderedDict
        )

        if input_types is not None:
            self._nodes.add(GraphSpace.Start)
            for i in input_types:
                self._types_registry[i][GraphSpace.Start] = None

    def __contains__(self, x: Any) -> bool:
        return x in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self):
        return iter(self._nodes)

    @property
    def algorithms(self):
        return [i.algorithm for i in self._nodes if hasattr(i, "algorithm")]

    def add(self, node: PipelineNode):
        if node in self._nodes:
            return

        self._nodes.add(node)

        self._types_registry[node.algorithm.output_type()][node] = None

    def remove(self, node: PipelineNode):
        if node not in self._nodes:
            return

        self._nodes.remove(node)

        types = self._types_registry[node.algorithm.output_type()]
        types.pop(node)

        if len(types) == 0:
            self._types_registry.pop(node.algorithm.output_type())

    @property
    def output_types(self) -> Set[type]:
        return self._types_registry.keys()

    def get_nodes_with_output(self, output_type: type) -> Sequence[PipelineNode]:
        if output_type not in self._types_registry:
            return set()

        return self._types_registry[output_type].keys()

    def get_lattest_node_with_output(self, output_type: type) -> PipelineNode:
        if output_type not in self._types_registry:
            return None

        last_node = next(reversed(self._types_registry[output_type]))
        return last_node


@nice_repr
class Pipeline:
    """Represents a sequence of algorithms.

    Each algorithm must have a `run` method declaring it's input and output type.
    The pipeline instance also receives the input and output types.
    """

    def __init__(
        self,
        algorithms: List[Algorithm],
        input_types: List[Type[SemanticType]],
        connection_mapping: Optional[Dict[Algorithm, List[Algorithm]]] = None,
    ) -> None:
        self.algorithms = algorithms
        self.input_types = input_types
        self.connection_mapping = connection_mapping

    def _run(self, *inputs):
        data: Dict[type, Any] = {}

        for i, t in zip(inputs, self.input_types):
            data[t] = i

        for algorithm in self.algorithms:
            args = build_input_args(algorithm, data)
            output = algorithm.run(**args)
            output_type = algorithm.output_type()
            data[output_type] = output

        return data[self.algorithms[-1].output_type()]

    def _run_with_connection(self, *inputs):
        input_data: Dict[type, Any] = {}
        algorithm_outputs: Dict[Algorithm, Any] = {}

        for i, t in zip(inputs, self.input_types):
            input_data[t] = i

        for algorithm in self.algorithms:
            connections: List[Algorithm] = self.connection_mapping[algorithm]
            # update data with the correct algorithm outputs
            data: Dict[type, Any] = OrderedDict()
            for connected_algorithm in connections:
                if connected_algorithm == GraphSpace.Start:
                    data.update(input_data)
                    continue
                algorithm_output = algorithm_outputs[connected_algorithm]
                data[connected_algorithm.output_type()] = algorithm_output

            args = build_input_args(algorithm, data)
            output = algorithm.run(**args)
            algorithm_outputs[algorithm] = output

        return algorithm_outputs[self.algorithms[-1]]

    def run(self, *inputs):
        if self.connection_mapping is None:
            return self._run(*inputs)

        return self._run_with_connection(*inputs)

    def send(self, msg: str, *args, **kwargs):
        found = False

        for step in self.algorithms:
            if hasattr(step, msg):
                getattr(step, msg)(*args, **kwargs)
                found = True
            elif hasattr(step, "send"):
                step.send(msg, *args, **kwargs)
                found = True

        if not found:
            warnings.warn(f"No step answered message {msg}.")


def _get_name(obj):
    return (
        obj.__name__ if hasattr(obj, "__name__") else getattr(obj, "__class__").__name__
    )


class PathContext:
    """Auxiliar class used for keep track of PipelineNodes used in a Pipeline and handled it types"""

    __slots__ = (
        "nodes_types",
        "path",
        "_pipeline_index",
        "_input_types",
        "_has_unique_connection_path",
    )

    def __init__(
        self,
        input_types: Optional[List[type]] = None,
        initial_path: Optional[Sequence[PipelineNode]] = None,
    ) -> None:
        self._input_types = input_types
        self.nodes_types = NodesTypes(input_types)
        self.path: List[PipelineNode] = []
        self._pipeline_index: Dict[PipelineNode, int] = {}
        self._has_unique_connection_path: bool = True

        if initial_path is not None:
            for symbol in initial_path:
                if symbol == GraphSpace.Start:
                    self.path.insert(0, symbol)
                    continue

                self.push(symbol)

    @property
    def has_unique_connection_path(self) -> bool:
        return self._has_unique_connection_path

    def push(self, node: PipelineNode):
        if node == GraphSpace.End:
            self.path.append(node)
            return

        nodes_types = self.nodes_types

        if self._has_unique_connection_path:
            # if we have two algorithm that produce the same output type then
            # there is no unique way to connect the pipeline algorithms
            for i_type in node.input_types:
                if len(nodes_types.get_nodes_with_output(i_type)) > 1:
                    self._has_unique_connection_path = False

        nodes_types.add(node)
        self._pipeline_index[node] = len(self.path)
        self.path.append(node)

    def pop(self) -> PipelineNode:
        if len(self.path) == 0:
            return None

        nodes_types = self.nodes_types
        node = self.path.pop()
        nodes_types.remove(node)
        self._pipeline_index.pop(node)

        # TODO: check if keep as unique connected when pop, for now keep track of this attribute by yourself when `push` and `pop` later
        # if not self._has_unique_connection_path:
        #    pass

        return node

    def top(self) -> PipelineNode:
        if len(self.path) == 0:
            return None

        return self.path[-1]

    def get_pipeline_index(self, node: PipelineNode) -> int:
        if node not in self._pipeline_index:
            raise Exception(f"Node {node} not found in this PathContext")

        return self._pipeline_index[node]

    def is_full_compatible_with(self, algorithm: Algorithm) -> bool:
        """
        Determines if the algorithm is compatible with a set of available input types in this context,
        i.e., if among those types we can find all the necessary inputs for this algorithm.
        """
        my_inputs = algorithm.input_types()
        context_available_types = self.nodes_types.output_types

        # check if we have the exact types, fast check
        if all(map(lambda x: x in context_available_types, my_inputs)):
            return True

        # slow check, since we can have a lot of available types in the context
        return algorithm.is_compatible_with(context_available_types)

    def __repr__(self) -> str:
        return f"<PathContext(input_types={[i.__name__ for i in self._input_types]})->path={[repr(i) for i in self.path]}>"


class PipelineSpace(GraphSpace):
    def __init__(self, graph: Graph, input_types: List[type]):
        super().__init__(graph, initializer=self._initialize)
        self.input_types = input_types

    def _initialize(self, item: PipelineNode, sampler):
        return item.sample(sampler)

    def nodes(self) -> Set[Type[Algorithm]]:
        """Returns a list of all algorithms (types) that exist in the graph.
        """
        return set(
            node.algorithm
            for node in self.graph.nodes
            if isinstance(node, PipelineNode)
        )

    def _generate_pipeline(
        self, path: Sequence[Algorithm], *, sampler: ISampler = None
    ) -> Pipeline:
        if sampler is None:
            sampler = Sampler()

        # create a context to know if this path has unique connections way
        context = PathContext(self.input_types)

        mapper: Dict[Algorithm, List[Algorithm]] = defaultdict(list)
        # keep track of algorithm whose outputs are used
        connected_algorithm: Set[Algorithm] = set()

        for algorithm in path:

            algorithm_input_types = algorithm.input_types()

            for i_type in algorithm_input_types:
                available_algorithm = list(
                    context.nodes_types.get_nodes_with_output(i_type)
                )

                if len(available_algorithm) > 1:
                    # wa sample over all available algorithms (current context) whose outputs are compatible with the current input
                    selected_algorithm = sampler.choice(available_algorithm)
                else:
                    selected_algorithm = available_algorithm[0]

                if selected_algorithm != GraphSpace.Start:
                    selected_algorithm = selected_algorithm.algorithm

                mapper[algorithm].append(selected_algorithm)
                connected_algorithm.add(selected_algorithm)

            # update context
            guaranteed_types = context.nodes_types.output_types
            p = PipelineNode(
                algorithm=algorithm,
                input_types=guaranteed_types,
                output_types=guaranteed_types | set([algorithm.output_type()]),
                registry=None,
                has_grammar=False,
            )
            context.push(p)

        # we always use the output of the last algorithm
        connected_algorithm.add(path[-1])

        unused_algorithm = set(path).difference(connected_algorithm)

        # prune all algorithm whose output isn't used
        if len(unused_algorithm) > 0:
            updated_path = []

            for algorithm in path:
                if algorithm in unused_algorithm:
                    mapper.pop(algorithm)
                    continue

                updated_path.append(algorithm)

            path = updated_path

        # if has unique connection way return the Pipeline
        if context.has_unique_connection_path:
            return Pipeline(path, input_types=self.input_types)

        # if don't have unique connection we pass the mapper dict
        return Pipeline(path, input_types=self.input_types, connection_mapping=mapper)

    def sample(self, *args, **kwargs):
        path = super().sample(*args, **kwargs)
        return self._generate_pipeline(path, *args, **kwargs)


class LazyPipelineSpace(PipelineSpace):
    """Like PipelineSpace but never build the full graph, it build the path at sampling time.
    """

    def __init__(
        self,
        input_types: List[type],
        output_type: type,
        registry: List[Algorithm],
        max_list_depth: int = 3,
        graph: Optional[Graph] = None,
    ):
        if graph is None:
            graph = Graph()

        super().__init__(graph, input_types)

        self._registry = registry

        # We start by enlarging the registry with all Seq[...] algorithms
        pool = set(registry)

        for algorithm in registry:
            for _ in range(max_list_depth):
                algorithm = make_seq_algorithm(algorithm)
                pool.add(algorithm)

        self._algorithms_registry = pool

        # We start by collecting all the possible input nodes,
        # those that can process a subset of the input_types
        initial_nodes: Set[PipelineNode] = set()
        input_types_set = set(input_types)

        for algorithm in pool:
            if not algorithm.is_compatible_with(input_types):
                continue

            initial_nodes.add(
                PipelineNode(
                    algorithm=algorithm,
                    input_types=input_types,
                    output_types=input_types_set | set([algorithm.output_type()]),
                    registry=registry,
                )
            )

        # raise no pipelines found if we can't connect any algorithm in the registry with the input
        if len(initial_nodes) == 0:
            raise TypeError("No pipelines can be found!")

        # precomputed initial nodes
        self._initial_nodes: List[Algorithm] = list(initial_nodes)
        self._output_type = output_type

    def _dfs_sampling(
        self, max_iterations: int, sampler: ISampler, context: PathContext
    ) -> Optional[GraphSpaceEnd]:
        if max_iterations <= 0:
            return None

        nodesTypes = context.nodes_types
        pool = self._algorithms_registry

        # We never want to apply the same exact algorithm twice, so we exclude already used algorithm
        valid_pool: Set[Algorithm] = pool.difference(nodesTypes.algorithms)

        guaranteed_types = nodesTypes.output_types

        available_algorithm: List[PipelineNode] = []

        for algorithm in valid_pool:
            # Check if the previous used algorithms has the required input types
            # This guarantee that we only add the algorithm when all the needed types are available
            if not context.is_full_compatible_with(algorithm):
                continue

            # We never want an algorithm that doesn't provide a novel output type...
            if (
                algorithm.output_type() in guaranteed_types
                and
                # ... unless it is an idempotent algorithm
                not (
                    len(algorithm.input_types()) == 1
                    and algorithm.output_type() == algorithm.input_types()[0]
                )  # we can rewrite this using logic but in this way we do not need adicional
                # check for indexing input_types() in 0
            ):
                continue

            # Since we have all the required input types across the path we connect only with the
            # lattest node in the path if this have one of the required types as output
            lattest_node_output_type = context.top().algorithm.output_type()
            if lattest_node_output_type not in set(algorithm.input_types()):
                continue

            p = PipelineNode(
                algorithm=algorithm,
                input_types=guaranteed_types,
                output_types=guaranteed_types | set([algorithm.output_type()]),
                registry=self._registry,
            )

            available_algorithm.append(p)

        # Now we check to see if the output can be connected with node last node in the path
        if issubclass(context.top().algorithm.output_type(), self._output_type):
            available_algorithm.append(GraphSpace.End)

        path_result = None

        while path_result is None and len(available_algorithm) > 0:

            node = sampler.choice(available_algorithm)

            self.graph.add_edge(context.top(), node)

            context_unique_connected = context.has_unique_connection_path
            context.push(node)

            if node == GraphSpace.End:
                return GraphSpace.End

            path_result = self._dfs_sampling(max_iterations - 1, sampler, context)

            if path_result == GraphSpace.End:
                return path_result

            # if we are here then the dfs take a wrong path that never connect with the End
            # we filter the used algorithm and try to sample again
            available_algorithm = list(filter(lambda x: x != node, available_algorithm))

            context.pop()
            context._has_unique_connection_path = context_unique_connected

            self.graph.remove_edge(context.top(), node)
            # prune node if isn't connected to a path that reach End
            if self.graph.out_degree(node) == 0:
                self.graph.remove_node(node)
        else:
            return None

    def _sample(self, symbol: List[Any], max_iterations: int, sampler: ISampler):
        context = PathContext(self.input_types, symbol)

        if max_iterations <= 0:
            raise ValueError("Reached maximum iterations")

        # sampling initial node
        node = sampler.choice(self._initial_nodes)
        self.graph.add_edge(GraphSpace.Start, node)
        context.push(node)

        if max_iterations <= 0:
            raise ValueError("Reached maximum iterations")

        self._dfs_sampling(max_iterations - 1, sampler, context)

        if context.top() != GraphSpace.End:
            if len(context.path) >= max_iterations:
                raise ValueError("Reached maximum iterations")

            raise ValueError("Cannot continue sampling. Graph is disconnected.")

        return [self.initializer(node, sampler=sampler) for node in context.path[1:-1]]


def _dfs_exploration(
    global_output_type: type,
    pool: Set[Algorithm],
    G: Graph,
    registry: List[Algorithm],
    context: PathContext,
):
    """Auxiliar method.
    Build a graph of algorithms making DFS exploration of the pipeline space.

    Every node in the graph corresponds to a <autogoal.grammar.ContextFreeGrammar> that
    generates an instance of a class with a `run` method.

    Each `run` method must declare input and output types in the form:

        def run(self, a: type_1, b: type_2, ...) -> type_n:
            # ...
    """

    nodesTypes = context.nodes_types

    # We never want to apply the same exact algorithm twice, so we exclude already used algorithm
    valid_pool: Set[Algorithm] = pool.difference(nodesTypes.algorithms)

    guaranteed_types = nodesTypes.output_types

    for algorithm in valid_pool:
        # Check if the previous used algorithms has the required input types
        # This guarantee that we only add the algorithm when all the needed types are available
        if not context.is_full_compatible_with(algorithm):
            continue

        # We never want an algorithm that doesn't provide a novel output type...
        if (
            algorithm.output_type() in guaranteed_types
            and
            # ... unless it is an idempotent algorithm
            not (
                len(algorithm.input_types()) == 1
                and algorithm.output_type() == algorithm.input_types()[0]
            )  # we can rewrite this using logic but in this way we do not need adicional
            # check for indexing input_types() in 0
        ):
            continue

        # Since we have all the required input types across the path we connect only with the
        # lattest node in the path if this have one of the required types as output
        lattest_node_output_type = context.top().algorithm.output_type()
        if lattest_node_output_type not in set(algorithm.input_types()):
            continue

        p = PipelineNode(
            algorithm=algorithm,
            input_types=guaranteed_types,
            output_types=guaranteed_types | set([algorithm.output_type()]),
            registry=registry,
        )

        G.add_edge(context.top(), p)

        # Now we check to see if this node is a possible output
        if issubclass(algorithm.output_type(), global_output_type):
            G.add_edge(p, GraphSpace.End)

        context_unique_connected = context.has_unique_connection_path
        context.push(p)
        _dfs_exploration(global_output_type, pool, G, registry, context)
        context.pop()
        context._has_unique_connection_path = context_unique_connected


def build_pipeline_graph(
    input_types: List[type],
    output_type: type,
    registry: List[Algorithm],
    max_list_depth: int = 3,
    is_lazy=False,
) -> PipelineSpace:
    """Build a graph of algorithms.

    Every node in the graph corresponds to a <autogoal.grammar.ContextFreeGrammar> that
    generates an instance of a class with a `run` method.

    Each `run` method must declare input and output types in the form:

        def run(self, a: type_1, b: type_2, ...) -> type_n:
            # ...
    """

    if not isinstance(input_types, (list, tuple)):
        input_types = [input_types]

    if is_lazy:
        return LazyPipelineSpace(input_types, output_type, registry, max_list_depth)

    # We start by enlarging the registry with all Seq[...] algorithms
    pool = set(registry)

    for algorithm in registry:
        for _ in range(max_list_depth):
            algorithm = make_seq_algorithm(algorithm)
            pool.add(algorithm)

    # We start by collecting all the possible input nodes,
    # those that can process a subset of the input_types
    initial_nodes: Set[PipelineNode] = set()
    input_types_set = set(input_types)

    for algorithm in pool:
        if not algorithm.is_compatible_with(input_types):
            continue

        initial_nodes.add(
            PipelineNode(
                algorithm=algorithm,
                input_types=input_types,
                output_types=input_types_set | set([algorithm.output_type()]),
                registry=registry,
            )
        )

    # raise no pipelines found if we can't connect any algorithm in the registry with the input
    if len(initial_nodes) == 0:
        raise TypeError("No pipelines can be found!")

    G = Graph()

    for node in initial_nodes:
        G.add_edge(GraphSpace.Start, node)
        context = PathContext(input_types)
        context.push(node)

        _dfs_exploration(output_type, pool, G, registry, context)

        # Now we check to see if this node is a possible output
        if issubclass(node.algorithm.output_type(), output_type):
            G.add_edge(node, GraphSpace.End)

    # Remove all nodes that are not connected to the end node
    try:
        reachable_from_end = set(
            nx.dfs_preorder_nodes(G.reverse(False), GraphSpace.End)
        )
        unreachable_nodes = set(G.nodes) - reachable_from_end
        G.remove_nodes_from(unreachable_nodes)
    except KeyError:
        raise TypeError("No pipelines can be found!")

    return PipelineSpace(G, input_types=input_types)


__all__ = [
    "AlgorithmBase",
    "Supervised",
    "Pipeline",
    "build_pipeline_graph",
    "algorithm",
]


# Finally, we run doctest in the module for easy testing of the functional API.

if __name__ == "__main__":
    import doctest

    doctest.testmod()
