from autogoal.experimental.hyperopt_search import format_hyperopt_args
from autogoal.grammar import ContinuousValue, CategoricalValue, DiscreteValue
from typing import Dict
from autogoal.sampling import Sampler
from autogoal.utils import nice_repr
from autogoal.search import PESearch
from autogoal.grammar import generate_cfg
from autogoal.grammar import (
    ContinuousValue,
    CategoricalValue,
    DiscreteValue,
    BooleanValue,
)
from autogoal.grammar import Union, Symbol
from hyperopt import fmin, tpe, rand, hp, space_eval
from autogoal.experimental.exact_sampler import ExactSampler
from autogoal.experimental.hyperopt_search import format_hyperopt_args, cfg_to_hp_space
from autogoal.kb import Document, Word, Stem, Seq, Sentence, algorithm
import hyperopt.pyll.stochastic
from numpy.random import RandomState


@nice_repr
class TextAlgorithm:
    def __init__(
        self, x: ContinuousValue(-10, 10), op: CategoricalValue("-", "+", "*", "/")
    ):
        self.x = x
        self.op = op
        pass

    def run(self, input: Sentence) -> Document:
        pass


@nice_repr
class StemWithDependanceAlgorithm:
    def __init__(
        self, y: DiscreteValue(0, 10), dependance: algorithm(Sentence, Document)
    ):
        self.y = y
        self.dependance = dependance
        pass

    def run(self, input: Word) -> Stem:
        pass


@nice_repr
class StemAlgorithm:
    def __init__(self, z: BooleanValue()):
        self.z = z
        pass

    def run(self, input: Word) -> Stem:
        pass


@nice_repr
class HigherStemAlgorithm:
    def __init__(self, dependance: algorithm(Word, Stem)):
        self.dependance = dependance
        pass

    def run(self, input: Seq[Word]) -> Seq[Stem]:
        pass


def equalSpaces(a, b):
    if type(a) != type(b):
        return False
    if type(a) == type(dict()):
        for key in {**a, **b}.keys():
            if str(a[key]) != str(b[key]):
                return False
        return True
    else:
        return str(a) == str(b)


def test_cfg_to_hp_space():
    cfg = generate_cfg(
        HigherStemAlgorithm,
        registry=[
            StemAlgorithm,
            TextAlgorithm,
            StemWithDependanceAlgorithm,
            HigherStemAlgorithm,
        ],
    )

    actual = cfg_to_hp_space(cfg)
    expected = cfg_to_hp_space(cfg)

    expected = {
        "Algorithm[[Word],Stem]": hp.choice(
            "Algorithm[[Word],Stem]",
            [
                {
                    "Algorithm[[Word],Stem]": Symbol("StemAlgorithm"),
                    "StemAlgorithm_z": hp.choice(
                        "StemAlgorithm_z",
                        [{"StemAlgorithm_z": False}, {"StemAlgorithm_z": True}],
                    ),
                },
                {
                    "Algorithm[[Word],Stem]": Symbol("StemWithDependanceAlgorithm"),
                    "StemWithDependanceAlgorithm_y": hp.quniform(
                        "StemWithDependanceAlgorithm_y", -0.5, 10.5, 1
                    ),
                    "Algorithm[[Sentence],Document]": hp.choice(
                        "Algorithm[[Sentence],Document]",
                        [
                            {
                                "Algorithm[[Sentence],Document]": Symbol(
                                    "TextAlgorithm"
                                ),
                                "TextAlgorithm_x": hp.uniform(
                                    "TextAlgorithm_x", -10, 10
                                ),
                                "TextAlgorithm_op": hp.choice(
                                    "TextAlgorithm_op",
                                    [
                                        {"TextAlgorithm_op": "-"},
                                        {"TextAlgorithm_op": "+"},
                                        {"TextAlgorithm_op": "*"},
                                        {"TextAlgorithm_op": "/"},
                                    ],
                                ),
                            }
                        ],
                    ),
                },
            ],
        )
    }

    assert equalSpaces(actual, expected)


def test_format_hyperopt_args():
    args = {
        "Algorithm[[Word],Stem]": {
            "Algorithm[[Sentence],Document]": {
                "Algorithm[[Sentence],Document]": Symbol(name="TextAlgorithm"),
                "TextAlgorithm_op": {"TextAlgorithm_op": "-"},
                "TextAlgorithm_x": -1.945047463027059,
            },
            "Algorithm[[Word],Stem]": Symbol(name="StemWithDependanceAlgorithm"),
            "StemWithDependanceAlgorithm_y": 1.0,
        }
    }
    expected = {
        "Algorithm[[Word],Stem]": Symbol(name="StemWithDependanceAlgorithm"),
        "StemWithDependanceAlgorithm_y": 1.0,
        "Algorithm[[Sentence],Document]": Symbol(name="TextAlgorithm"),
        "TextAlgorithm_op": "-",
        "TextAlgorithm_x": -1.945047463027059,
        "StemWithDependanceAlgorithm_y": 1.0,
    }
    formatted_args = format_hyperopt_args(args)
    assert expected == formatted_args


def test_cfg_to_solution():
    cfg = generate_cfg(
        HigherStemAlgorithm,
        registry=[
            StemAlgorithm,
            TextAlgorithm,
            StemWithDependanceAlgorithm,
            HigherStemAlgorithm,
        ],
    )
    hp_space = cfg_to_hp_space(cfg)
    sample = hyperopt.pyll.stochastic.sample(hp_space, rng=RandomState(1))
    sample = format_hyperopt_args(sample)
    solution = cfg.sample(sampler=ExactSampler(sample))
    expected = HigherStemAlgorithm(
        StemWithDependanceAlgorithm(10.0, TextAlgorithm(8.651147186773176, "+"))
    )
    assert repr(solution) == repr(expected)
