from typing import Dict, cast, List
import math
import time
from hyperopt import fmin, tpe, rand, hp, space_eval, STATUS_OK, STATUS_FAIL
from hyperopt.early_stop import no_progress_loss

from autogoal.experimental.exact_sampler import ExactSampler
from autogoal.grammar import Symbol, generate_cfg
from autogoal.grammar._cfg import (
    Callable,
    Distribution,
    OneOf,
    SubsetOf,
    ContextFreeGrammar,
)
from autogoal.kb._algorithm import PipelineSpace
from autogoal.utils import RestrictedWorkerByJoin, Min, Gb, Sec
from autogoal.search._base import MultiLogger

ALGORITHM_CHOICE_HANDLE = "__CHOICE__"


def cfg_to_hp_space(cfg: ContextFreeGrammar):
    """
    Transforms a `ContextFreeGrammar` in a hyperopt search space
    """
    return _cfg_to_hp_space(cfg, None, None)


def _cfg_to_hp_space(cfg, symbol=None, choice_ref=None):
    start: Symbol = cfg._start if symbol is None else symbol
    node_name = start.name
    grammar_node = cfg[start]
    if type(grammar_node) == Distribution:
        ditribution_name = grammar_node.name
        if ditribution_name == "discrete":
            min_value = grammar_node.parameters["min"] - 0.5
            max_value = grammar_node.parameters["max"] + 0.5
            return hp.quniform(node_name, min_value, max_value, 1)
        elif ditribution_name == "continuous":
            min_value = grammar_node.parameters["min"]
            max_value = grammar_node.parameters["max"]
            return hp.uniform(node_name, min_value, max_value)
        elif ditribution_name == "categorical":
            return hp.choice(
                node_name,
                [{node_name: option} for option in grammar_node.parameters["options"]],
            )
        elif ditribution_name == "boolean":
            return hp.choice(
                node_name, [{node_name: option} for option in [False, True]],
            )

    elif type(grammar_node) == OneOf:
        hp_options = []
        for option in grammar_node.options:
            hp_options.append(_cfg_to_hp_space(cfg, option, node_name))
        return hp.choice(node_name, hp_options)
    elif type(grammar_node) == Callable:
        parameters = {}
        if choice_ref is not None:
            parameters[choice_ref] = start
        for key, value in grammar_node.parameters.items():
            if type(value) == Symbol:
                parameters[value.name] = _cfg_to_hp_space(cfg, value)
            else:
                parameters[key] = _cfg_to_hp_space(cfg, value)
        return parameters
    elif type(grammar_node) == SubsetOf:
        raise NotImplementedError('Hyperopt\'s "subset of" is not yet implemented')


def format_hyperopt_args(args: Dict) -> Dict:
    """
    Transforms arguments in hyperopt's format to a model compatible with ExactSampler
    """
    new_args = {}
    for key, value in args.items():
        if type(value) == type(dict()):
            new_args = {**new_args, **format_hyperopt_args(value)}
        else:
            if key.startswith(ALGORITHM_CHOICE_HANDLE):
                # anything would do, we're just interested in the key here.
                new_args[value] = True
            else:
                new_args[key] = value
    new_args["End"] = True
    return new_args


def pipeline_space_to_hp_space(pipeline_space, registry):
    """
    Transforms a `PipelineSpace` into a hyperopt search space
    """
    return _pipeline_space_to_hp_space(pipeline_space, registry, None, [])


def _pipeline_space_to_hp_space(pipeline_space, registry, start=None, path: List = []):

    if start == pipeline_space.End:
        return None
    graph = pipeline_space.graph
    start_node = start if start is not None else pipeline_space.Start
    node_choices = []
    node_path = path.copy() + [start_node]
    for sub_node in graph[start_node].keys():
        node_choices.append(
            _pipeline_space_to_hp_space(pipeline_space, registry, sub_node, node_path)
        )
    node_choices = list(filter(lambda choice: choice is not None, node_choices))
    if start_node != pipeline_space.Start:
        node_space = cfg_to_hp_space(
            generate_cfg(start_node.algorithm, registry=registry)
        )
        if type(node_space) != type(dict()):
            raise RuntimeError("Unexpected output type for final CFG")
        else:
            node_space = cast(Dict, node_space)
        choice_path = f"{ALGORITHM_CHOICE_HANDLE}{get_path_string(path)}"
        node_space[choice_path] = start_node.algorithm.__name__
    else:
        node_space = {}
    if len(node_choices) > 0:
        path_str = f"{ALGORITHM_CHOICE_HANDLE}{get_path_string(node_path)}"
        node_space[path_str] = hp.choice(
            path_str, [{path_str: option} for option in node_choices]
        )
    return node_space


def get_path_string(path):
    path_strings = [
        repr(node_step.algorithm.__name__)
        if hasattr(node_step, "algorithm")
        else repr(node_step)
        for node_step in path
    ]
    return "->".join(path_strings)
