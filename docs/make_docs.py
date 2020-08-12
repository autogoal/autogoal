# Convert examples in this folder to their corresponding .md files in docs/examples

import re
import inspect
import textwrap
import datetime
import yaml
from pathlib import Path


def hide(line):
    return ":hide:" in line


def build_examples():
    current = Path(__file__)
    folder = current.parent

    for fname in folder.rglob("*.py"):
        if fname.name.startswith("_"):
            continue
        if fname.name == current.name:
            continue

        process(fname)


class Markdown:
    def __init__(self, content):
        while content:
            if not content[0].strip():
                content.pop(0)
            else:
                break

        while content:
            if not content[-1].strip():
                content.pop()
            else:
                break

        self.content = content

    def print(self, fp):
        for line in self.content:
            if line.startswith("# "):
                fp.write(line[2:])
            else:
                fp.write("\n")

        fp.write("\n")


class Python(Markdown):
    def print(self, fp):
        if not self.content:
            return

        fp.write("```python\n")

        for line in self.content:
            fp.write(line)

        fp.write("```\n\n")


def process(fname: Path):
    print(fname)

    content = []

    with fname.open("r") as fp:
        current = []
        state = "markdown"

        for line in fp:
            if hide(line):
                continue

            if line.startswith("#"):
                if state == "python":
                    if current:
                        content.append(Python(current))
                        current = []
                    state = "markdown"
                current.append(line)
            else:
                if state == "markdown":
                    if current:
                        content.append(Markdown(current))
                        current = []
                    state = "python"
                current.append(line)

        if current:
            if state == "markdown":
                content.append(Markdown(current))
            else:
                content.append(Python(current))

    output = fname.parent / (fname.name[:-3] + ".md")

    with output.open("w") as fp:
        for c in content:
            c.print(fp)


def build_api():
    import autogoal
    import autogoal.contrib
    import autogoal.datasets
    import autogoal.grammar
    import autogoal.kb
    import autogoal.ml
    import autogoal.sampling
    import autogoal.search

    index = []
    generate(autogoal, index)
    lines = yaml.dump(index)

    with open(Path(__file__).parent.parent / "mkdocs-base.yml", "r") as fr:
        with open(Path(__file__).parent.parent / "mkdocs.yml", "w") as fw:
            for line in fr:
                fw.write(line)

            fw.write("    - API:\n")
            for line in lines.splitlines():
                fw.write(f"        {line}\n")


def generate(module, index, visited=set()):
    name = module.__name__

    if name in visited:
        return

    visited.add(name)
    print(name)

    path = Path(__file__).parent / "api" / (name + ".md")
    submodules = inspect.getmembers(
        module,
        lambda m: inspect.ismodule(m)
        and m.__name__.startswith("autogoal")
        and not "._" in m.__name__,
    )
    classes = inspect.getmembers(
        module,
        lambda m: inspect.isclass(m)
        and m.__module__.startswith(module.__name__)
        and not m.__name__.startswith("_"),
    )
    functions = inspect.getmembers(
        module,
        lambda m: inspect.isfunction(m)
        and m.__module__.startswith(module.__name__)
        and not m.__name__.startswith("_"),
    )

    members_index = [{"Index": f"api/{name}.md"}]
    index.append({name: members_index})

    with open(path, "w") as fp:
        generate_module(module, name, fp)

        if submodules:
            fp.write("\n## Submodules\n\n")

            for _, submodule in submodules:
                fp.write(f"* [{submodule.__name__}](../{submodule.__name__}/)\n")
                generate(submodule, index)

        if classes:
            fp.write("\n## Classes\n\n")

            for _, clss in classes:
                generate_class(clss, name, fp)
                members_index.append({clss.__name__: f"api/{name}.{clss.__name__}.md"})

        if functions:
            fp.write("\n## Functions\n\n")

            for _, func in functions:
                generate_func(func, name, fp)
                members_index.append({func.__name__: f"api/{name}.{func.__name__}.md"})


def format_param(p: inspect.Parameter) -> str:
    if p.default != p.empty:
        return f"{p.name}={repr(p.default)}"

    if p.kind == inspect.Parameter.VAR_POSITIONAL:
        return f"*{p.name}"

    if p.kind == inspect.Parameter.VAR_KEYWORD:
        return f"**{p.name}"

    return f"{p.name}"


def format_signature(obj, name=None) -> str:
    if name is None:
        name = obj.__name__

    signature = inspect.signature(obj)
    params = ", ".join(format_param(p) for p in signature.parameters.values())
    return f"{name}({params})"


def generate_class(clss, name, fp):
    print(name, clss)
    doc = inspect.getdoc(clss)

    fp.write(f"### [`{clss.__name__}`](../{name}.{clss.__name__})\n")
    if doc:
        fp.write(f"> {doc.splitlines()[0]}\n\n")

    fp = open(Path(__file__).parent / "api" / f"{name}.{clss.__name__}.md", "w")
    fp.write(f"# `{name}.{clss.__name__}`\n\n")

    src = inspect.getsourcefile(clss)
    if src:
        line = inspect.getsourcelines(clss)[1]
        src = src.replace(
            "/usr/lib/python3/dist-packages/",
            "https://github.com/autogal/autogoal/blob/main/",
        )
        src_link = f"> [📝]({src}#L{line})\n"
        fp.write(src_link)

    fp.write(f"> `{format_signature(clss.__init__, clss.__name__)}`\n\n")

    if doc:
        fp.write(doc)
        fp.write("\n")

    members = inspect.getmembers(
        clss, lambda m: inspect.isfunction(m) and not m.__name__.startswith("_")
    )

    for _, member in members:
        generate_func(member, name, fp, indent="###", new_file=False)


def generate_func(func, name, fp, indent="###", new_file=True):
    print(name, func)
    doc = inspect.getdoc(func)

    if new_file:
        fp.write(f"{indent} [`{func.__name__}`](../{name}.{func.__name__})\n")
        if doc:
            fp.write(f"> {doc.splitlines()[0]}\n\n")

        fp = open(Path(__file__).parent / "api" / f"{name}.{func.__name__}.md", "w")
        fp.write(f"# `{name}.{func.__name__}`\n\n")
    else:
        fp.write(f"{indent} `{func.__name__}`\n\n")

    src = inspect.getsourcefile(func)
    if src:
        line = inspect.getsourcelines(func)[1]
        src = src.replace(
            "/usr/lib/python3/dist-packages/",
            "https://github.com/autogoal/autogoal/blob/main/",
        )
        src_link = f"> [📝]({src}#L{line})\n"
        fp.write(src_link)

    fp.write(f"> `{format_signature(func)}`\n\n")

    if doc:
        fp.write(doc)
        fp.write("\n")

    if new_file:
        fp.close()


def generate_module(module, name, fp):
    doc = module.__doc__
    fp.write(f"# `{module.__name__}`\n")

    if doc is not None:
        fp.write(doc)


def build_schemas():
    from autogoal.kb._data import draw_data_hierarchy

    draw_data_hierarchy(str(Path(__file__).parent / "guide" / "datatypes"))


def make_algorithms_table():
    from autogoal.contrib import find_classes

    all_classes = find_classes()

    with open(Path(__file__).parent / "guide" / "algorithms.md", "w") as fp:
        fp.write(textwrap.dedent(
            """
            |Algorithm|Dependencies|Input|Output|
            |--|--|--|--|
            """
        ))

        for clss in all_classes:
            print(clss)
            signature = inspect.signature(clss.run)
            dependency = clss.__module__.split('.')[2]

            if dependency.startswith('_'):
                dependency = ""

            fp.write(f"| {clss.__name__} | {dependency} | {signature.parameters['input'].annotation} | {signature.return_annotation} | \n")


if __name__ == "__main__":
    build_examples()
    build_schemas()
    make_algorithms_table()
    build_api()
