from collections import namedtuple
import inspect
import abc
import types
import warnings
from autogoal.datasets import Dataset, SimpleDataset
import yaml
from typing import Any, Dict, List, Set, Tuple, Type
from pathlib import Path
import types
import os
import shutil
import re

import networkx as nx
from autogoal.utils import nice_repr
from autogoal.grammar import Graph, GraphSpace, generate_cfg
from autogoal.kb._semantics import SemanticType, Seq
from autogoal.utils import (
    AlgorithmConfig,
    get_contrib,
    loads,
    dumps,
)
import dill as pickle


def algorithm(*annotations, exceptions: List[str] = None):
    from autogoal.grammar import Union, Symbol

    *inputs, output = annotations

    def match(cls):
        if not hasattr(cls, "run"):
            return False

        if exceptions is not None:
            for exc in exceptions:
                if exc in cls.__name__ or exc in cls.__module__:
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


class Algorithm(abc.ABC):
    """Represents an abstract algorithm with a run method.

    Provides introspection for the expected semantic input and output types.
    Users should inherit from `AlgorithmBase` instead of this class.
    """

    @abc.abstractclassmethod
    def input_types(cls) -> Tuple[type]:
        """Returns an ordered list of the expected semantic input types of the `run` method."""
        pass

    @abc.abstractclassmethod
    def input_args(cls) -> Tuple[str]:
        """Returns an ordered tuple of the names of the arguments in the `run` method."""
        pass

    @abc.abstractclassmethod
    def output_type(cls) -> type:
        """Returns an ordered list of the expected semantic output type of the `run` method."""
        pass

    @abc.abstractmethod
    def run(self, *args):
        """Executes the algorithm."""
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


class AlgorithmBase(Algorithm):
    """Represents an algorithm,

    Automatically implements the input and output introspection methods using the `inspect` module.
    Users inheriting from this class must provide type annotations in the `run` method.
    """

    @classmethod
    def init_input_types(cls) -> Tuple[type]:
        return tuple(
            param.annotation
            for name, param in inspect.signature(cls.__init__).parameters.items()
            if name != "self"
        )

    @classmethod
    def get_inner_signature(cls) -> inspect.Signature:
        return inspect.signature(cls.__init__)

    @classmethod
    def is_upscalable(cls) -> bool:
        return True

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

    def save(self, path: Path):
        """
        Serializes the Algorithm instance.
        """
        self._save_info(path)
        self.save_model(path)

    def _save_info(self, path: Path):
        params = [
            name
            for name in inspect.signature(self.__init__).parameters
            if name != "self"
        ]
        values = [getattr(self, param, None) for param in params]
        parameters = {params[i]: values[i] for i in range(len(params))}

        for pkey in parameters.keys():
            if isinstance(parameters[pkey], AlgorithmBase):
                param_path = path / str(pkey)
                os.mkdir(param_path)
                p_config = parameters[pkey].save(param_path)
                parameters[pkey] = p_config

        module = f"'{self.__module__}.{self.__class__.__name__}'"
        name = self.__class__.__name__
        config = AlgorithmConfig(name, module, parameters)
        config.to_yaml(path)
        return config

    def save_model(self, path: Path) -> "None":
        with open(path / "model.bin", "wb") as fd:
            pickle.Pickler(fd).dump(self)

    @classmethod
    def load(self, path: Path) -> "AlgorithmBase":
        """
        Deserializes an Algorithm instance.
        """
        return self.load_model(path)

    @classmethod
    def _get_info(self, path: Path) -> "AlgorithmConfig":
        return AlgorithmConfig.from_yaml(path)

    @classmethod
    def load_model(self, path: Path):
        with open(path / "model.bin", "rb") as fd:

            algorithm = pickle.Unpickler(fd).load()

            if not isinstance(algorithm, AlgorithmBase):
                raise ValueError(
                    "The serialized file does not contain an AlgorithmBase instance."
                )

            return algorithm


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
            result[name] = values[type].load_all_data()
        except KeyError:
            for key in values:
                if issubclass(key, type):
                    result[name] = values[key].load_all_data()
                    break
            else:
                raise TypeError(f"Cannot find compatible input value for {type}")

    return result


@nice_repr
class Pipeline:
    """Represents a sequence of algorithms.

    Each algorithm must have a `run` method declaring it's input and output type.
    The pipeline instance also receives the input and output types.
    """

    def __init__(
        self, algorithms: List[Algorithm], input_types: List[Type[SemanticType]]
    ) -> None:
        self.algorithms = algorithms
        self.input_types = input_types

    def run(self, *inputs):
        data = {}

        for i, t in zip(inputs, self.input_types):
            data[t] = SimpleDataset(t, i)

        for algorithm in self.algorithms:
            args = build_input_args(algorithm, data)
            output = algorithm.run(**args)
            output_type = algorithm.output_type()
            data[output_type] = SimpleDataset(output_type, output)

        return data[self.algorithms[-1].output_type()].load_all_data()

    def send(self, msg: str, *args, **kwargs):
        found = False

        for step in self.algorithms:
            if hasattr(step, msg):
                getattr(step, msg)(*args, **kwargs)
                found = True
            elif hasattr(step, "send"):
                step.send(msg, *args, **kwargs)  # type: ignore
                found = True

        if not found:
            warnings.warn(f"No step answered message {msg}.")

    def save_algorithms(self, path: Path):
        save_path = path / "algorithms"
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.mkdir(save_path)

        algorithms = []
        contribs = set()
        info = {}

        for i, algorithm in enumerate(self.algorithms):
            contribs.add(get_contrib(algorithm.__class__))
            algorithm_path = save_path / str(i)
            os.mkdir(algorithm_path)
            algorithm.save(algorithm_path)
            algorithm_class = f"'{algorithm.__module__}.{algorithm.__class__.__name__}'"
            algorithms.append(algorithm_class)

        info["algorithms"] = algorithms
        info["inputs"] = dumps(self.input_types)

        with open(path / "algorithms.yml", "w") as fd:
            yaml.dump(info, fd)

        return contribs

    def serialize(self, use_dill=False):
        return dumps(self, use_dill)

    @staticmethod
    def deserialize(data, use_dill=False):
        return loads(data, use_dill)

    @staticmethod
    def load_algorithms(path: Path):
        """
        Load pipeline algorithms list from given path
        """
        with open(path / "algorithms.yml", "r") as fd:
            stored_data = yaml.safe_load(fd)

        from autogoal_contrib import find_classes

        autogoal_algorithms = find_classes()

        algorithms = []

        algorithm_clases = []

        for i, algorithm in enumerate(stored_data.get("algorithms")):

            # Patch for when an algorithm is a generated seq algorithm.
            # If that is the case then we need to know the depth of the sequence algorithms
            # in order to build the original algorithm
            inner_algorithm, seq_depth = find_nested_seq_algorithm_and_depth(algorithm)

            algorithm = inner_algorithm if seq_depth > 0 else algorithm

            for cls in autogoal_algorithms:
                if algorithm in object.__str__(cls):
                    algorithm_clases.append(cls)

                    loaded_algorithm = cls.load(path / "algorithms" / str(i))
                    while seq_depth > 0:
                        loaded_algorithm = make_seq_algorithm(loaded_algorithm)()
                        seq_depth -= 1
                    algorithms.append(loaded_algorithm)

        inputs = loads(stored_data.get("inputs"))
        return (algorithms, inputs)


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

    is_algorithm_instance = isinstance(algorithm, AlgorithmBase)

    if not (
        algorithm.__class__.is_upscalable()
        if is_algorithm_instance
        else algorithm.is_upscalable()
    ):
        return None

    output_type = algorithm.output_type()

    name = f"SeqAlgorithm[{algorithm.__class__.__name__ if is_algorithm_instance else algorithm.__name__}]"

    def init_method(self, *args, **kwargs):
        self.inner = algorithm if is_algorithm_instance else algorithm(*args, **kwargs)

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


def find_nested_seq_algorithm_and_depth(s):
    # Find all occurrences of the pattern
    matches = re.findall(r"abc\.SeqAlgorithm\[", s)

    # The depth is the number of occurrences
    depth = len(matches)

    # Find the innermost string by removing the outer layers
    inner_string = re.sub(r"'", "", s)
    for _ in range(depth):
        inner_string = re.sub(r"abc\.SeqAlgorithm\[", "", inner_string, count=1)
        inner_string = re.sub(r"\]$", "", inner_string, count=1)

    return inner_string, depth


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


class PipelineNode:
    def __init__(self, algorithm, input_types, output_types, registry=None) -> None:
        self.algorithm = algorithm
        self.input_types = set(input_types)
        self.output_types = set(output_types)
        self.grammar = generate_cfg(self.algorithm, registry=registry)

    def sample(self, sampler):
        return self.grammar.sample(sampler=sampler)

    @property
    def __name__(self):
        return self.algorithm.__name__

    def __eq__(self, o: object) -> bool:
        return isinstance(o, PipelineNode) and all(
            [
                o.algorithm == self.algorithm,
                o.input_types == self.input_types,
            ]
        )

    def __repr__(self) -> str:
        return f"<PipelineNode(algorithm={self.algorithm.__name__},input_types={[i.__name__ for i in self.input_types]},output_types={[o.__name__ for o in self.output_types]})>"

    def __hash__(self) -> int:
        return hash(repr(self))


class PipelineSpace(GraphSpace):
    def __init__(self, graph: Graph, input_types):
        super().__init__(graph, initializer=self._initialize)
        self.input_types = input_types

    def _initialize(self, item: PipelineNode, sampler):
        return item.sample(sampler)

    def nodes(self) -> Set[Type[Algorithm]]:
        """Returns a list of all algorithms (types) that exist in the graph."""
        return set(
            node.algorithm
            for node in self.graph.nodes
            if isinstance(node, PipelineNode)
        )

    def sample(self, *args, **kwargs):
        path = super().sample(*args, **kwargs)
        return Pipeline(path, input_types=self.input_types)


def build_pipeline_graph(
    input_types: List[type],
    output_type: type,
    registry: List[Algorithm],
    max_list_depth: int = 3,
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

    # We start by enlarging the registry with all Seq[...] algorithms

    pool = set(registry)

    for algorithm in registry:
        for _ in range(max_list_depth):
            algorithm = make_seq_algorithm(algorithm)
            if not algorithm is None:
                pool.add(algorithm)

    # For building the graph, we'll keep at each node the guaranteed output types

    # We start by collecting all the possible input nodes,
    # those that can process a subset of the input_types
    open_nodes: List[PipelineNode] = []

    for algorithm in pool:
        if not algorithm.is_compatible_with(input_types):
            continue

        open_nodes.append(
            PipelineNode(
                algorithm=algorithm,
                input_types=input_types,
                output_types=set(input_types) | set([algorithm.output_type()]),
                registry=registry,
            )
        )

    G = Graph()

    for node in open_nodes:
        G.add_edge(GraphSpace.Start, node)

    # We'll make a BFS exploration of the pipeline space.
    # For every open node we will add to the graph every node to which it can connect.
    closed_nodes = set()

    while open_nodes:
        node = open_nodes.pop(0)

        # These are the types that are available at this node
        guaranteed_types = node.output_types

        # The node's output type
        node_output_type = node.algorithm.output_type()

        # Here are all the algorithms that could be added new at this point in the graph
        for algorithm in pool:
            if not algorithm.is_compatible_with(guaranteed_types):
                continue

            # We never want to apply the same exact algorithm twice
            if algorithm == node.algorithm:
                continue

            # And we never want an algorithm that doesn't provide a novel output type...
            if (
                algorithm.output_type() in guaranteed_types
                and
                # ... unless it is an idempotent algorithm
                tuple([algorithm.output_type()]) != algorithm.input_types()
            ):
                continue

            # BUG: this validation ensures no redundant nodes are added.
            #      The downside is that it prevents pipelines that need two algorithms
            #      to generate the input of another one.

            # And we do not want to ignore the last node's output type
            is_using_last_output = False
            for input_type in algorithm.input_types():
                if issubclass(node_output_type, input_type):
                    is_using_last_output = True
                    break
            if not is_using_last_output:
                continue

            p = PipelineNode(
                algorithm=algorithm,
                input_types=guaranteed_types,
                output_types=guaranteed_types | set([algorithm.output_type()]),
                registry=registry,
            )

            G.add_edge(node, p)

            if p not in closed_nodes and p not in open_nodes:
                open_nodes.append(p)

        # Now we check to see if this node is a possible output
        if issubclass(node.algorithm.output_type(), output_type):
            G.add_edge(node, GraphSpace.End)

        closed_nodes.add(node)

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
    "Pipeline",
    "build_pipeline_graph",
    "algorithm",
]


# Finally, we run doctest in the module for easy testing of the functional API.

if __name__ == "__main__":
    import doctest

    doctest.testmod()
