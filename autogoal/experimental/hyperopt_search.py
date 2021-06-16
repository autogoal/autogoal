from typing import Dict
from autogoal.grammar import Symbol
from autogoal.grammar._cfg import Callable, Distribution, OneOf, SubsetOf
from hyperopt import fmin, tpe, rand, hp, space_eval


def cfg_to_hp_space(cfg, symbol=None, choice_ref=None):
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
            hp_options.append(cfg_to_hp_space(cfg, option, node_name))
        return hp.choice(node_name, hp_options)
    elif type(grammar_node) == Callable:
        parameters = {}
        if choice_ref is not None:
            parameters[choice_ref] = start
        for key, value in grammar_node.parameters.items():
            if type(value) == Symbol:
                parameters[value.name] = cfg_to_hp_space(cfg, value)
            else:
                parameters[key] = cfg_to_hp_space(cfg, value)
        return parameters
    elif type(grammar_node) == SubsetOf:
        raise NotImplementedError('Hyperopt\'s "subset of" is not yet implemented')


def format_hyperopt_args(args: Dict, parent_arg: str = None) -> Dict:
    """
    This method is responsible for formatting hyperopt's args in order to be compatible
    with an ExactSampler.
    """
    new_args = {}
    for key, value in args.items():
        if type(value) == type(dict()):
            new_args = {**new_args, **format_hyperopt_args(args[key], key)}
        else:
            new_args[key] = value
    return new_args
