#%%
import pydot
import inspect
from IPython.display import display_svg

from autogoal.kb import _semantics

classes = inspect.getmembers(_semantics, lambda c: inspect.isclass(c) and issubclass(c, _semantics.SemanticType))

output_file = "semantic_types"
graph = pydot.Dot(direction="TB")

for name, clss in classes:
    graph.add_node(pydot.Node(name))

parents = {}

for name, clss in classes:
    parent = _semantics.SemanticType
    parent_name = "SemanticType"

    for name2, base in classes:
        if name2 == name:
            continue

        if issubclass(clss, base) and issubclass(base, parent):
            parent = base
            parent_name = name2

    parents[name] = parent_name

for name, parent in parents.items():
    if name == parent:
        continue

    graph.add_edge(pydot.Edge(parent, name))

graph.write(output_file + ".svg", format="svg")
graph.write(output_file + ".png", format="png")
graph.write(output_file + ".pdf", format="pdf")


display_svg(graph.create_svg().decode("utf8"), raw=True)

# %%
