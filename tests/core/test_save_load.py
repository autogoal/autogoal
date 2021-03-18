from io import BytesIO
from pickle import Pickler, Unpickler

from autogoal.datasets import dummy
from autogoal.grammar import CategoricalValue, DiscreteValue, generate_cfg
from autogoal.kb import (
    CategoricalVector,
    List,
    MatrixContinuousDense,
    Tuple,
    build_composite_list,
    build_composite_tuple,
)
from autogoal.ml import AutoML
from autogoal.search import RandomSearch
from autogoal.utils import nice_repr


@nice_repr
class A:
    def __init__(self, x: DiscreteValue(-10, 10), y: DiscreteValue(-10, 10)):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return repr(self) == repr(other)


def fn(a: A):
    return a.x ** 2 + a.y ** 2


def test_search_is_replayable_from_grammar():
    grammar = generate_cfg(A)
    search = RandomSearch(generator_fn=grammar, fitness_fn=fn)
    best, _ = search.run(1)

    sampler = best.sampler_.replay()
    best_clone = grammar(sampler)

    assert best == best_clone


def fn2(sampler):
    return sampler.discrete(0, 10)


def test_search_is_replayable_from_fitness_no_multiprocessing():
    search = RandomSearch(fitness_fn=fn2, evaluation_timeout=0, memory_limit=0)
    best, best_fn = search.run(10)

    sampler = best.sampler_.replay()

    assert best is sampler
    assert sampler._history != []
    assert best_fn == fn2(sampler)


@nice_repr
class DummyAlgorithm:
    def __init__(self, x: CategoricalValue("A", "B", "C")):
        self.x = x

    def train(self):
        pass

    def eval(self):
        pass

    def run(
        self, input: Tuple(MatrixContinuousDense(), CategoricalVector())
    ) -> CategoricalVector():
        X, y = input
        return y


def test_automl_save_load():
    X, y = dummy.generate(seed=0)
    automl = AutoML(search_iterations=3, registry=[DummyAlgorithm])
    automl.fit(X, y)
    pipe = automl.best_pipeline_

    fp = BytesIO()

    automl.save(fp)
    fp.seek(0)

    automl2 = AutoML.load(fp)
    pipe2 = automl2.best_pipeline_

    assert repr(pipe) == repr(pipe2)


def test_save_load_tuple():
    TupleClass = build_composite_tuple(
        1,
        input_type=Tuple(MatrixContinuousDense(), CategoricalVector()),
        output_type=Tuple(MatrixContinuousDense(), CategoricalVector()),
    )
    algorithm = TupleClass(DummyAlgorithm)

    fp = BytesIO()

    Pickler(fp).dump(algorithm)
    fp.seek(0)

    algorithm2 = Unpickler(fp).load()

    assert repr(algorithm) == repr(algorithm2)


def test_save_load_list():
    ListClass = build_composite_list(
        input_type=Tuple(MatrixContinuousDense(), CategoricalVector()),
        output_type=List(CategoricalVector()),
    )
    algorithm = ListClass(DummyAlgorithm)

    fp = BytesIO()

    Pickler(fp).dump(algorithm)
    fp.seek(0)

    algorithm2 = Unpickler(fp).load()

    assert repr(algorithm) == repr(algorithm2)
