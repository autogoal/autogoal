from autogoal.grammar import ContinuousValue, CategoricalValue, DiscreteValue
from typing import Dict, List, cast
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
from autogoal.experimental.hyperopt_search import (
    format_hyperopt_args,
    cfg_to_hp_space,
    pipeline_space_to_hp_space,
)
from autogoal.kb import Seq, algorithm, Postag, Pipeline
from autogoal.kb._algorithm import make_seq_algorithm
import hyperopt.pyll.stochastic
from numpy.random import RandomState
from autogoal.kb import build_pipeline_graph, AlgorithmBase
from autogoal.kb import SemanticType
import networkx as nx
import matplotlib.pyplot as plt
from hyperopt.pyll import Apply


class AType(SemanticType):
    ...


class BType(SemanticType):
    ...


class CType(SemanticType):
    ...


class DType(SemanticType):
    ...


@nice_repr
class AD(AlgorithmBase):
    def __init__(
        self, x: ContinuousValue(-10, 10), op: CategoricalValue("-", "+", "*", "/")
    ):
        self.x = x
        self.op = op
        pass

    def run(self, input: AType) -> DType:
        pass


@nice_repr
class BCWithDependance(AlgorithmBase):
    def __init__(self, y: DiscreteValue(0, 10), dependance: algorithm(AType, DType)):
        self.y = y
        self.dependance = dependance
        pass

    def run(self, input: BType) -> CType:
        pass


@nice_repr
class BC(AlgorithmBase):
    def __init__(self, z: BooleanValue()):
        self.z = z
        pass

    def run(self, input: BType) -> CType:
        pass


@nice_repr
class HigherBCAlgorithm(AlgorithmBase):
    def __init__(self, dependance: algorithm(BType, CType)):
        self.dependance = dependance
        pass

    def run(self, input: Seq[BType]) -> Seq[CType]:
        pass


@nice_repr
class CSeqA(AlgorithmBase):
    def __init__(self, a: DiscreteValue(-10, -1)):
        self.a = a
        pass

    def run(self, input: CType) -> Seq[AType]:
        pass


@nice_repr
class CD(AlgorithmBase):
    def __init__(self, a: DiscreteValue(-10, -1)):
        self.a = a
        pass

    def run(self, input: CType) -> DType:
        pass


def equalSpaces(a, b):
    # BUG: this is actually not working as the same space can be stringified in different ways
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
        HigherBCAlgorithm, registry=[BC, AD, BCWithDependance, HigherBCAlgorithm,],
    )

    actual = cfg_to_hp_space(cfg)

    expected = {
        "Algorithm[[Btype],Ctype]": hp.choice(
            "Algorithm[[Btype],Ctype]",
            [
                {
                    "Algorithm[[Btype],Ctype]": Symbol("BC"),
                    "BC_z": hp.choice("BC_z", [{"BC_z": False}, {"BC_z": True}],),
                },
                {
                    "Algorithm[[Btype],Ctype]": Symbol("BCWithDependance"),
                    "BCWithDependance_y": hp.quniform(
                        "BCWithDependance_y", -0.5, 10.5, 1
                    ),
                    "Algorithm[[Atype],Dtype]": hp.choice(
                        "Algorithm[[Atype],Dtype]",
                        [
                            {
                                "Algorithm[[Atype],Dtype]": Symbol("AD"),
                                "AD_x": hp.uniform("AD_x", -10, 10),
                                "AD_op": hp.choice(
                                    "AD_op",
                                    [
                                        {"AD_op": "-"},
                                        {"AD_op": "+"},
                                        {"AD_op": "*"},
                                        {"AD_op": "/"},
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


def test_pipeline_space_to_hp_space():
    registry = [AD, BCWithDependance, BC, HigherBCAlgorithm, CSeqA, CD]
    g = build_pipeline_graph(Seq[BType], Seq[DType], registry,)

    actual = pipeline_space_to_hp_space(g, registry)

    seq_CD_space_higher_BC_space = {
        "__CHOICE__Start()->HigherBCAlgorithm": "SeqAlgorithm[CD]",
        "SeqAlgorithm[CD]_a": hp.quniform("BCWithDependance_y", -10.5, 10.5, 1),
    }

    higher_BC_space = {
        "__CHOICE__Start()": "HigherBCAlgorithm",
        "Algorithm[[Btype],Ctype]": hp.choice(
            "Algorithm[[Btype],Ctype]",
            [
                {
                    "Algorithm[[Btype],Ctype]": Symbol("BC"),
                    "BC_z": hp.choice("BC_z", [{"BC_z": False}, {"BC_z": True}],),
                },
                {
                    "Algorithm[[Btype],Ctype]": Symbol("BCWithDependance"),
                    "BCWithDependance_y": hp.quniform(
                        "BCWithDependance_y", -0.5, 10.5, 1
                    ),
                    "Algorithm[[Atype],Dtype]": hp.choice(
                        "Algorithm[[Atype],Dtype]",
                        [
                            {
                                "Algorithm[[Atype],Dtype]": Symbol("AD"),
                                "AD_x": hp.uniform("AD_x", -10, 10),
                                "AD_op": hp.choice(
                                    "AD_op",
                                    [
                                        {"AD_op": "-"},
                                        {"AD_op": "+"},
                                        {"AD_op": "*"},
                                        {"AD_op": "/"},
                                    ],
                                ),
                            }
                        ],
                    ),
                },
            ],
        ),
        "__CHOICE__Start()->HigherBCAlgorithm": hp.choice(
            "__CHOICE__Start()->HigherBCAlgorithm", [seq_CD_space_higher_BC_space]
        ),
    }

    seq_CD_space_BC_dep_space = {
        "__CHOICE__Start()->SeqAlgorithm[BCWithDependance]": "SeqAlgorithm[CD]",
        "SeqAlgorithm[CD]_a": hp.quniform("BCWithDependance_y", -10.5, 10.5, 1),
    }

    seq_BC_dep_space = {
        "__CHOICE__Start()": "SeqAlgorithm[BCWithDependance]",
        "SeqAlgorithm[BCWithDependance]_y": hp.quniform(
            "SeqAlgorithm[BCWithDependance]_y", -0.5, 10.5, 1
        ),
        "Algorithm[[Atype],Dtype]": hp.choice(
            "Algorithm[[Atype],Dtype]",
            [
                {
                    "Algorithm[[Atype],Dtype]": Symbol("AD"),
                    "AD_x": hp.uniform("AD_x", -10, 10),
                    "AD_op": hp.choice(
                        "AD_op",
                        [
                            {"AD_op": "-"},
                            {"AD_op": "+"},
                            {"AD_op": "*"},
                            {"AD_op": "/"},
                        ],
                    ),
                }
            ],
        ),
        "__CHOICE__Start()->SeqAlgorithm[BCWithDependance]": hp.choice(
            "__CHOICE__Start()->SeqAlgorithm[BCWithDependance]",
            [seq_CD_space_BC_dep_space],
        ),
    }

    seq_CD_space_BC_space = {
        "__CHOICE__Start()->SeqAlgorithm[BC]": "SeqAlgorithm[CD]",
        "SeqAlgorithm[CD]_a": hp.quniform("SeqAlgorithm[CD]_a", -10.5, 10.5, 1),
    }

    seq_BC_space = {
        "__CHOICE__Start()": "SeqAlgorithm[BC]",
        "SeqAlgorithm[BC]_z": hp.choice(
            "SeqAlgorithm[BC]_z",
            [{"SeqAlgorithm[BC]_z": False}, {"SeqAlgorithm[BC]_z": True}],
        ),
        "__CHOICE__Start()->SeqAlgorithm[BC]": hp.choice(
            "__CHOICE__Start()->SeqAlgorithm[BC]", [seq_CD_space_BC_space]
        ),
    }

    expected = {
        "__CHOICE__Start()": hp.choice(
            "__CHOICE__Start()", [higher_BC_space, seq_BC_dep_space, seq_BC_space]
        )
    }

    assert equalSpaces(actual, expected)


def test_format_hyperopt_args():
    args = {
        "Algorithm[[Btype],Ctype]": {
            "Algorithm[[Atype],Dtype]": {
                "Algorithm[[Atype],Dtype]": Symbol(name="AD"),
                "AD_op": {"AD_op": "-"},
                "AD_x": -1.945047463027059,
            },
            "Algorithm[[Btype],Ctype]": Symbol(name="BCWithDependance"),
            "BCWithDependance_y": 1.0,
        }
    }
    expected = {
        "Algorithm[[Btype],Ctype]": Symbol(name="BCWithDependance"),
        "BCWithDependance_y": 1.0,
        "Algorithm[[Atype],Dtype]": Symbol(name="AD"),
        "AD_op": "-",
        "AD_x": -1.945047463027059,
        "BCWithDependance_y": 1.0,
    }
    formatted_args = format_hyperopt_args(args)
    assert expected == formatted_args


def test_cfg_sample_to_solution():
    cfg = generate_cfg(
        HigherBCAlgorithm, registry=[BC, AD, BCWithDependance, HigherBCAlgorithm,],
    )
    sample = {
        "Algorithm[[Btype],Ctype]": Symbol(name="BCWithDependance"),
        "BCWithDependance_y": 1.0,
        "Algorithm[[Atype],Dtype]": Symbol(name="AD"),
        "AD_op": "-",
        "AD_x": -1.945047463027059,
        "BCWithDependance_y": 1.0,
    }
    solution = cfg.sample(sampler=ExactSampler(sample))
    expected = HigherBCAlgorithm(BCWithDependance(10.0, AD(8.651147186773176, "+")))
    assert repr(solution) == repr(expected)


def test_pipeline_sample_to_solution():
    registry = [AD, BCWithDependance, BC, HigherBCAlgorithm, CSeqA, CD]
    g = build_pipeline_graph(Seq[BType], Seq[DType], registry,)
    hp_formatted_sample = {
        "AD_op": "/",
        "End": True,
        "AD_x": -3.9533485473632046,
        "Algorithm[[AType],DType]": Symbol(name="AD"),
        "Algorithm[[BType],CType]": Symbol(name="BCWithDependance"),
        "BCWithDependance_y": -0.0,
        "HigherBCAlgorithm": True,
        "SeqAlgorithm[CD]_a": -1.0,
        "SeqAlgorithm[CD]": True,
    }
    solution = g.sample(sampler=ExactSampler(hp_formatted_sample))
    seq_cd = make_seq_algorithm(CD)
    expected = Pipeline(
        [
            HigherBCAlgorithm(BCWithDependance(-0.0, AD(-3.9533485473632046, "/"))),
            seq_cd(-1.0),
        ],
        [Seq[BType]],
    )
    assert repr(solution) == repr(expected)
