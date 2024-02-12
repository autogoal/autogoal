import inspect
from pathlib import Path
from typing import List


def get_requirements(cls):
    print(inspect.getfile(cls))
    path = Path(inspect.getfile(cls))
    requiements = path.parent / "requirements.txt"
    if requiements.exists():
        return requiements.read_text()
    return None


def generate_requirements(list: List):
    with open("requirements.txt", "w") as fd:
        for algorithm in list:
            data = get_requirements(algorithm)
            if data is not None:
                fd.writelines(data)


def get_contrib(cls):
    path = Path(inspect.getfile(cls))
    return path.parent.name


def generate_installer(path: Path, list: List):
    with open(path / "contribs.sh", "w") as fd:
        fd.writelines("#!/bin/bash\n")
        base = "poetry install\n"
        for contrib in list:
            base += f" -E {contrib}"
        fd.writelines(base)


def resolve_contrib(contrib: str):
    if not contrib.startswith("autogoal_"):
        contrib = f"autogoal_{contrib}"
    return __import__(contrib)
