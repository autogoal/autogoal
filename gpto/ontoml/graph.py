# coding: utf8

from pathlib import Path
from pprint import pprint as pp

import networkx as nx
import owlready2 as owl

from .ontology import onto
onto.load()


def _get_name(ind):
    return ind.name.split(".")[-1]


def build_graph(input_data, output_data):
    G = nx.DiGraph()

    for ind in onto['Algorithm'].instances():
        if ind.hasInput is None or ind.hasOutput is None:
            continue

        if ind.implementedIn is None:
            continue

        G.add_node(_get_name(ind))

    nodes = list(G.nodes)

    G.add_nodes_from(['START', 'END'])

    for u in nodes:
        uind = onto[u]

        if uind.hasInput in input_data.isCoercibleTo:
            G.add_edge('START', u, label='canConnect')

        if output_data in uind.hasOutput.isCoercibleTo:
            G.add_edge(u, 'END', label='canConnect')

        for v in nodes:
            if u == v:
                continue

            vind = onto[v]

            if vind in uind.canConnect:
                G.add_edge(u, v, label='canConnect')

    reachable_from_start = set(nx.dfs_preorder_nodes(G, 'START'))
    reachable_from_end = set(nx.dfs_preorder_nodes(G.reverse(False), 'END'))
    reachable = reachable_from_start & reachable_from_end
    all_nodes = set(G.nodes)
    nodes_to_remove = all_nodes - reachable
    G.remove_nodes_from(nodes_to_remove)

    remove_cycles(G, 'START')

    return G


def remove_cycles(G, inputs):
    try:
        while True:
            cycle = nx.find_cycle(G, inputs)
            G.remove_edge(*cycle[-1])
    except nx.NetworkXNoCycle:
        pass


def enum_paths(input_data,  output_data, max_steps=5):
    G = build_graph(input_data, output_data)

    for path in nx.all_simple_paths(G, 'START', 'END', max_steps):
        print(path)


def _find_path(current_path, steps, outputData):
    last = current_path[-1]

    if steps == 0:
        if last.hasOutput == outputData:
            yield current_path
    else:
        for ind in last.canConnect:
            new_path = current_path + [ind]
            yield from _find_path(new_path, steps-1, outputData)


if __name__ == "__main__":
    import sys
    # G = build_graph(onto[sys.argv[1]], onto[sys.argv[2]])
    enum_paths(onto[sys.argv[1]], onto[sys.argv[2]])

