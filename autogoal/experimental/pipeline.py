import inspect
import abc
import types
from typing import Any, Dict, List, Tuple

import networkx as nx
from autogoal.utils import nice_repr
from autogoal.grammar import Graph, GraphSpace, generate_cfg
from autogoal.kb import List as _List, conforms, DataType


class Supervised(DataType):
    def __init__(self, internal, **tags):
        super().__init__(**tags)
        self.internal = internal

    def __conforms__(self, other):
        return isinstance(other, Supervised) and conforms(self.internal, other.internal)

    def __name__(self):
        return f"Supervised({self.internal.__name__})"

   
class Algorithm(abc.ABC):
    """Represents an abstract algorithm with a run method.

    Provides introspection for the expected semantic input and output types.
    Users should inherit from `AlgorithmBase` instead of this class.
    """
    @abc.abstractclassmethod
    def input_types(cls) -> Tuple[type]:
        """Returns an ordered list of the expected semantic input types of the `run` method.
        """
        pass

    @abc.abstractclassmethod
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
                if conforms(having, needed):
                    break
            else:
                return False

        return True


class AlgorithmBase(Algorithm):
    """Represents an algorithm. 

    Automatically implements the input and output introspection methods using the `inspect` module.
    Users inheriting from this class must provide type annotations in the `run` method.
    """
    @classmethod
    def input_types(cls) -> Tuple[type]:
        return tuple(param.annotation for name, param in inspect.signature(cls.run).parameters.items() if name != 'self')

    @classmethod
    def output_type(cls) -> type:
        return inspect.signature(cls.run).return_annotation



def _build_input_args(algorithm:Algorithm, values:Dict[type,Any]):
    """Buils the correct input mapping for `algorithm` using the provided `values` mapping types to objects.

    The input can be a class that inherits from `Algorithm` or an instance of such a class.

    >>> class A(AlgorithmBase):
    ...    def run(self, a:int, b:str):
    ...        pass
    >>> values = { str:"hello", float:3.0, int:42 }
    >>> _build_input_args(A, values)
    {'a': 42, 'b': 'hello'}
    >>> _build_input_args(A(), values)
    {'a': 42, 'b': 'hello'}

    """
    parameters = [p for p in inspect.signature(algorithm.run).parameters if p != "self"]
    result = {}

    for name, type in zip(parameters, algorithm.input_types()):
        result[name] = values[type]

    return result


@nice_repr
class Pipeline:
    """Represents a sequence of algorithms.

    Each algorithm must have a `run` method declaring it's input and output type.
    The pipeline instance also receives the input and output types.
    """
    def __init__(self, algorithms:List[Algorithm]) -> None:
        self.algorithms = algorithms

    def run(self, *inputs):
        data = {}

        for i,t in zip(inputs, self.algorithms[0].input_types()):
            data[t] = i

        for algorithm in self.algorithms:
            args = _build_input_args(algorithm, data)
            output = algorithm.run(**args)
            output_type = algorithm.output_type()
            data[output_type] = output

        return data[self.algorithms[-1].output_type()]


def _make_list_algorithm(algorithm: Algorithm):
    """Lift an algorithm with input types T1, T2, Tn to a meta-algorithm with types List[T1], List[T2], ...

    The generated class correctly defines the input and output types.

    >>> class A(AlgorithmBase):
    ...     def __init__(self, alpha):
    ...         self.alpha = 0.5
    ...     def run(self, x:int, y:str) -> float:
    ...         return self.alpha * (x + len(y))
    ...     def __repr__(self):
    ...         return f"A({self.alpha})"
    >>> B = _make_list_algorithm(A)
    >>> b = B(0.5)
    >>> b
    ListAlgorithm[A(0.5)]
    >>> b.run([1, 2], ["A", "BC"])
    [1.0, 2.0]
    >>> B.input_types()
    (List(<class 'int'>), List(<class 'str'>))
    >>> b.output_type()
    List(<class 'float'>)
    
    """
    
    output_type = algorithm.output_type()

    name = f"ListAlgorithm[{algorithm.__name__}]"

    def init_method(self, *args, **kwargs):
        self.inner = algorithm(*args, **kwargs)

    def run_method(self, *args) -> _List(output_type):
        return [self.inner.run(*xs) for xs in zip(*args)]

    def repr_method(self):
        return f"ListAlgorithm[{repr(self.inner)}]"

    def getattr_method(self, attr):
        return getattr(self.inner, attr)

    @classmethod
    def input_types_method(cls):
        return tuple(_List(t) for t in algorithm.input_types())

    @classmethod
    def output_types_method(cls):
        return _List(algorithm.output_type())

    def body(ns):
        ns["__init__"] = init_method
        ns["run"] = run_method
        ns["__repr__"] = repr_method
        ns["__getattr__"] = getattr_method
        ns["input_types"] = input_types_method
        ns["output_type"] = output_types_method

    return types.new_class(name=name, bases=(Algorithm,), exec_body=body)


class PipelineNode:
    def __init__(self, algorithm, input_types, output_types) -> None:
        self.algorithm = algorithm
        self.input_types = set(input_types)
        self.output_types = set(output_types)
        self.grammar = generate_cfg(self.algorithm)

    def sample(self, sampler):
        return self.grammar.sample(sampler=sampler)

    @property
    def __name__(self):
        return self.algorithm.__name__

    def __eq__(self, o: object) -> bool:
        return isinstance(o, PipelineNode) and all([
            o.algorithm == self.algorithm,
            o.input_types == self.input_types,
        ])

    def __repr__(self) -> str:
        return f"<PipelineNode(algorithm={self.algorithm.__name__},input_types={[i.__name__ for i in self.input_types]},output_types={[o.__name__ for o in self.output_types]})>"

    def __hash__(self) -> int:
        return hash(repr(self))


class PipelineSpace(GraphSpace):
    def __init__(self, graph: Graph, input_types):
        super().__init__(graph, initializer=self.initialize)
        self.input_types = input_types

    def initialize(self, item: PipelineNode, sampler):
        return item.sample(sampler)

    def sample(self, *args, **kwargs):
        path = super().sample(*args, **kwargs)
        return Pipeline(path)


def build_pipeline_graph(input_types, output_type, registry, max_list_depth: int=3):
    """Build a graph of algorithms.

    Every node in the graph corresponds to a <autogoal.grammar.ContextFreeGrammar> that
    generates an instance of a class with a `run` method.

    Each `run` method must declare input and output types in the form:

        def run(self, a: type_1, b: type_2, ...) -> type_n:
            # ...
    """
    
    # We start by enlarging the registry with all List[...] algorithms

    pool = set(registry)

    for algorithm in registry:
        for _ in range(max_list_depth):            
            algorithm = _make_list_algorithm(algorithm)
            pool.add(algorithm)

    # For building the graph, we'll keep at each node the guaranteed output types

    # We start by collecting all the possible input nodes,
    # those that can process a subset of the input_types
    open_nodes: List[PipelineNode] = []

    for algorithm in registry:
        if not algorithm.is_compatible_with(input_types):
            continue

        open_nodes.append(PipelineNode(
            algorithm = algorithm,
            input_types = input_types,
            output_types = set(input_types) | set([algorithm.output_type()])
        ))

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

        # Here are all the algorithms that could be added new at this point in the graph
        for algorithm in registry:
            if not algorithm.is_compatible_with(guaranteed_types):
                continue

            # We never want to apply the same exact algorithm twice
            if algorithm == node.algorithm:
                continue

            # And we never want an algorithm that doesn't provide a novel output type...
            if (algorithm.output_type() in guaranteed_types and 
                # ... unless it is an idempotent algorithm
                [algorithm.output_type()] != algorithm.input_types()
            ) :
                continue

            p = PipelineNode(
                algorithm = algorithm,
                input_types = guaranteed_types,
                output_types = guaranteed_types | set([algorithm.output_type()])
            )

            G.add_edge(node, p)

            if p not in closed_nodes:
                open_nodes.append(p)

        # Now we check to see if this node is a possible output
        if conforms(node.algorithm.output_type(), output_type):
            G.add_edge(node, GraphSpace.End)

        closed_nodes.add(node)

    # Remove all nodes that are not connected to the end node    
    reachable_from_end = set(nx.dfs_preorder_nodes(G.reverse(False), GraphSpace.End))
    unreachable_nodes = set(G.nodes) - reachable_from_end
    G.remove_nodes_from(unreachable_nodes)

    return PipelineSpace(G, input_types=input_types)


### TESTS

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
    def run(self, t1:T1) -> T2:
        pass

@nice_repr
class T2_T3(AlgorithmBase):
    def run(self, t2:T2) -> T3:
        pass

@nice_repr
class T3_T4(AlgorithmBase):
    def run(self, t3:T3) -> T4:
        pass

@nice_repr
class T2_T3_Supervised(AlgorithmBase):
    def run(self, x:T2, y:Supervised(T3)) -> T3:
        pass

def test_build_pipeline_with_two_algorithms():
    pipeline_builder = build_pipeline_graph(input_types=(T1,), output_type=T3, registry=[T1_T2, T2_T3])
    pipeline = pipeline_builder.sample()
    
    assert repr(pipeline.algorithms) == "[T1_T2(), T2_T3()]"


def test_build_pipeline_with_supervised():
    pipeline_builder = build_pipeline_graph(input_types=(T1,Supervised(T3),), output_type=T3, registry=[T1_T2, T2_T3_Supervised])
    pipeline = pipeline_builder.sample()
    
    assert repr(pipeline.algorithms) == "[T1_T2(), T2_T3_Supervised()]"


def test_build_pipeline_has_no_extra_nodes():
    pipeline_builder = build_pipeline_graph(input_types=(T1,), output_type=T3, registry=[T1_T2, T2_T3, T3_T4])
    
    print(pipeline_builder.graph.nodes)

    assert "T3_T4" not in [node.algorithm.__name__ for node in pipeline_builder.graph if hasattr(node, "algorithm")]

    
class Float2Str(AlgorithmBase):
    def run(self, a:float) -> str:
        return str(a)

class StrToInt(AlgorithmBase):
    def run(self, b:str) -> int:
        return len(b)

def test_when_pipeline_has_two_algorithms_then_passes_the_output():
    pipeline = Pipeline([Float2Str(), StrToInt()])
    result = pipeline.run(3.0)
    assert result == 3


class TwoInputAlgorithm(AlgorithmBase):
    def run(self, a:int, b:str) -> int:
        return a * len(b)

def test_when_pipeline_step_has_more_that_one_input_then_all_arguments_are_passed():
    pipeline = Pipeline([TwoInputAlgorithm()])
    assert pipeline.run(3, "hello world") == 33


def test_when_pipeline_second_step_receives_two_input_one_from_previous_and_one_from_origin():
    pipeline = Pipeline([StrToInt(), TwoInputAlgorithm()])
    result = pipeline.run("hello world")
    assert result == 121


from autogoal.kb import MatrixContinuous, CategoricalVector
from autogoal.contrib import find_classes


def test_build_real_pipeline():
    pipeline = build_pipeline_graph(input_types=(MatrixContinuous(), Supervised(CategoricalVector)), output_type=CategoricalVector(), registry=find_classes())
