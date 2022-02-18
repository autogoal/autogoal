from io import BytesIO
from pickle import loads, dumps

from autogoal.datasets import dummy
from autogoal.grammar import generate_cfg, DiscreteValue, CategoricalValue
from autogoal.kb import *
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


def fn(a: A, **kwargs):
    return a.x ** 2 + a.y ** 2


def test_save_seq():
    t1 = Seq[Word]
    t2 = loads(dumps(t1))

    assert id(t1) == id(t2)


def test_save_tensor():
    t1 = Tensor[2, Continuous, Dense]
    t2 = loads(dumps(t1))

    assert id(t1) == id(t2)


def test_search_is_replayable_from_grammar():
    grammar = generate_cfg(A)
    search = RandomSearch(generator_fn=grammar, fitness_fn=fn)
    best, _ = search.run(1)

    sampler = best.sampler_.replay()
    best_clone = grammar(sampler)

    assert best == best_clone


def fn2(sampler, **kwargs):
    return sampler.discrete(0, 10)


def test_search_is_replayable_from_fitness_no_multiprocessing():
    search = RandomSearch(fitness_fn=fn2, evaluation_timeout=0, memory_limit=0)
    best, best_fn = search.run(10)

    sampler = best.sampler_.replay()

    assert best is sampler
    assert sampler._history != []
    assert best_fn == fn2(sampler)


@nice_repr
class DummyAlgorithm(AlgorithmBase):
    def __init__(self, x: CategoricalValue("A", "B", "C")):
        self.x = x

    def train(self):
        pass

    def eval(self):
        pass

    def run(
        self, x: MatrixContinuousDense, y: Supervised[VectorCategorical]
    ) -> VectorCategorical:
        return y


def test_automl_save_load():
    # X, y = dummy.generate(seed=0)
    # automl = AutoML(
    #     input=(MatrixContinuousDense, Supervised[VectorCategorical]),
    #     output=VectorCategorical,
    #     search_iterations=3,
    #     registry=[DummyAlgorithm],
    # )

    # automl.fit(X, y)
    # pipe = automl.best_pipeline_

    # fp = BytesIO()

    # automl.save(fp)
    # fp.seek(0)

    # automl2 = AutoML.load(fp)
    # pipe2 = automl2.best_pipeline_

    # assert repr(pipe) == repr(pipe2)
    pass
