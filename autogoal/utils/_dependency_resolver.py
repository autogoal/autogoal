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
    print(inspect.getfile(cls))
    path = Path(inspect.getfile(cls))
    return path.parent.name

def generate_installer(path: Path, list: List):
    with open(path / "contribs.sh", "w") as fd:
        fd.writelines("#!/bin/bash\n")
        if 'keras' in list:
            fd.writelines("conda install -y tensorflow-gpu==2.1.0 && pip install tensorflow-addons==0.9.1\n")
        if 'transformers' in list:
            fd.writelines("pip install torch==1.10.1 torchvision==0.11.2\n")
        base = "poetry install"
        for contrib in list:
            base += f" -E {contrib}"
        fd.writelines(base)
