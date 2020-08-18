from autogoal.search import RandomSearch
from autogoal.grammar import generate_cfg, Discrete
from autogoal.utils import nice_repr
from autogoal.ml import AutoML
from autogoal.datasets import dummy


@nice_repr
class A:
    def __init__(self, x: Discrete(-10, 10), y: Discrete(-10, 10)):
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


from autogoal.kb import MatrixContinuousDense, CategoricalVector, Tuple
from autogoal.grammar import Categorical
from autogoal.utils import nice_repr
from io import BytesIO


@nice_repr
class DummyAlgorithm:
    def __init__(self, x:Categorical('A', 'B', 'C')):
        self.x = x

    def train(self):
        pass

    def eval(self):
        pass

    def run(self, input:Tuple(MatrixContinuousDense(), CategoricalVector())) -> CategoricalVector():
        X,y = input
        return y


def test_automl_save_load():
    X,y = dummy.load(seed=0)
    automl = AutoML(search_iterations=3, registry=[DummyAlgorithm])
    automl.fit(X, y)
    pipe = automl.best_pipeline_

    fp = BytesIO()

    automl.save_pipeline(fp)
    fp.seek(0)

    automl2 = AutoML(registry=[DummyAlgorithm])
    automl2.load_pipeline(fp)
    pipe2 = automl2.best_pipeline_

    assert repr(pipe) == repr(pipe2)
