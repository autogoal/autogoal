# coding: utf8

import argparse
import collections
import os
import sys
from collections import OrderedDict
from pprint import pprint

import networkx as nx
import yaml
from matplotlib import pyplot as plt
from networkx.readwrite import json_graph

from .ontology import onto


def get_grammar(input_type, output_type, depth=2, include=[], exclude=[]):
    datatypes = set(onto.Data.instances()) - {input_type, output_type}
    algorithms = [
        i
        for i in onto.Algorithm.instances()
        if i.hasInput is not None and i.hasOutput is not None
    ]

    if include:
        algorithms = [i for i in algorithms if isinstance(i, tuple(include))]
    if exclude:
        algorithms = [i for i in algorithms if not isinstance(i, tuple(exclude))]

    pipelines = _build_paths(input_type, output_type, datatypes, algorithms, {}, depth)
    # pprint(pipelines)

    productions = {}
    _build_grammar(
        pipelines,
        "Pipeline",
        productions,
        {},
        input_type,
        output_type,
        collections.Counter(),
    )

    return productions


def _build_grammar(
    tree, symbol, productions, inverse_productions, input_type, output_type, used_names
):
    direct = tree.get("direct", None)
    indirect = tree.get("indirect", [])

    symbol_productions = set()

    if direct:
        symbol_productions.add(
            _add_productions(direct, productions, inverse_productions, used_names)
        )

    for inds in indirect:
        step1, step2 = inds["step1"], inds["step2"]
        step1_symbol = _build_grammar(
            step1,
            None,
            productions,
            inverse_productions,
            input_type,
            inds["through"],
            used_names,
        )
        step2_symbol = _build_grammar(
            step2,
            None,
            productions,
            inverse_productions,
            inds["through"],
            output_type,
            used_names,
        )
        symbol_productions.add("%s %s" % (step1_symbol, step2_symbol))

    name = symbol or "%s_%s" % (
        input_type.name.split(".")[-1],
        output_type.name.split(".")[-1],
    )

    if name in used_names:
        lhs = name + str(used_names[name] + 1)
    else:
        lhs = name

    rhs = " | ".join(sorted(symbol_productions))

    if rhs in inverse_productions:
        return inverse_productions[rhs]

    if not indirect and symbol != "Pipeline":
        return list(symbol_productions)[0]

    productions[lhs] = rhs
    inverse_productions[rhs] = lhs

    used_names[name] += 1

    return lhs


def _add_productions(tree, productions, inverse_productions, used_names):
    name = tree["id"].name.split(".")[-1]
    children = tree.get("children")

    if not children:
        # TODO: Generate production for an instance algorithm
        if name in productions:
            return name

        parameters = onto[name].hasParameter
        productions[name] = " ".join(param.name.split(".")[-1] for param in parameters)

        for param in parameters:
            param_name = param.name.split(".")[-1]

            if isinstance(param, onto.StringHyperParameter):
                productions[param_name] = " | ".join(param.hasStringValues)
            if isinstance(param, onto.BooleanHyperParameter):
                productions[param_name] = "yes | no"
            if isinstance(param, onto.ContinuousHyperParameter):
                productions[param_name] = "f({},{})".format(
                    param.hasMinFloatValue, param.hasMaxFloatValue
                )
            if isinstance(param, onto.DiscreteHyperParameter):
                productions[param_name] = "i({},{})".format(
                    param.hasMinIntValue, param.hasMaxIntValue
                )

        return name
    else:
        # It's a set of algorithms
        rhs = " | ".join(
            sorted(
                _add_productions(c, productions, inverse_productions, used_names)
                for c in children
            )
        )

        if rhs in inverse_productions:
            lhs = inverse_productions[rhs]
        else:
            if name in used_names:
                lhs = name + str(used_names[name] + 1)
            else:
                lhs = name

            used_names[name] += 1
            productions[lhs] = rhs
            inverse_productions[rhs] = lhs

        return lhs


def _organize_tree(algorithms):
    tree = nx.DiGraph()

    for alg in algorithms:
        while alg != onto.Algorithm:
            parent = alg.is_a[0]
            tree.add_edge(parent, alg)
            alg = parent

    json_tree = json_graph.tree_data(tree, onto.Algorithm)
    return _simplify_tree(json_tree)


def _simplify_tree(json_tree):
    children = json_tree.get("children")

    if not children:
        return json_tree

    if len(children) == 1:
        return _simplify_tree(children[0])

    json_tree["children"] = [_simplify_tree(c) for c in children]
    return json_tree


def _build_paths(input_type, output_type, datatypes, algorithms, cache, depth):
    if (input_type, output_type, depth) in cache:
        return cache[(input_type, output_type, depth)]

    direct_paths = []

    for algorithm in algorithms:
        if (
            algorithm.hasInput in input_type.isCoercibleTo
            and output_type in algorithm.hasOutput.isCoercibleTo
        ):
            direct_paths.append(algorithm)

    direct_paths = _organize_tree(direct_paths) if direct_paths else []
    result = dict(direct=direct_paths) if direct_paths else None

    if depth == 0:
        cache[(input_type, output_type, depth)] = result
        return result

    if not datatypes:
        cache[(input_type, output_type, depth)] = result
        return result

    indirect_paths = []

    for data in sorted(datatypes, key=str):
        step1 = _build_paths(
            input_type, data, datatypes - {data}, algorithms, cache, depth - 1
        )
        step2 = _build_paths(
            data, output_type, datatypes - {data}, algorithms, cache, depth - 1
        )

        if step1 and step2:
            indirect_paths.append(dict(through=data, step1=step1, step2=step2))

    if indirect_paths:
        result = dict(direct=direct_paths, indirect=indirect_paths)

    cache[(input_type, output_type, depth)] = result
    return result


if __name__ == "__main__":
    import autogoal.ontology._nn as nn
    import autogoal.ontology._generated._keras as keras
    import pprint

    namespace = dict(vars(nn))
    namespace.update(vars(keras))

    grammar = nn.NeuralNetwork.generate_grammar()

    neural_network: nn.NeuralNetwork = grammar.sample(**namespace)
    print(neural_network)

    neural_network.compile((32,))
    print(neural_network.model.summary())

    # import sys

    # parser = argparse.ArgumentParser()
    # parser.add_argument("input")
    # parser.add_argument("output")
    # parser.add_argument("--include", nargs="+", default=[])
    # parser.add_argument("--exclude", nargs="+", default=[])
    # parser.add_argument("--depth", type=int, default=2)

    # args = parser.parse_args()

    # input_type = onto[args.input]
    # output_type = onto[args.output]
    # depth = args.depth
    # include = [onto[i] for i in args.include]
    # exclude = [onto[i] for i in args.exclude]

    # pprint(
    #     get_grammar(
    #         input_type, output_type, depth=depth, include=include, exclude=exclude
    #     )
    # )

