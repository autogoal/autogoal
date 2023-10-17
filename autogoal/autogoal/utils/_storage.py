from typing import List
import yaml
import dill
import pickle
from pathlib import Path
import shutil
import os


def dumps(data: object, use_dill=False) -> str:
    data = dill.dumps(data) if use_dill else pickle.dumps(data)
    return decode(data)


def decode(data: bytes) -> str:
    return data.decode("latin1")


def loads(data: str, use_dill=False):
    raw_data = data.encode("latin1")
    return dill.loads(raw_data) if use_dill else pickle.loads(raw_data)


def encode(code: str):
    return code.encode("latin1")


def ensure_directory(path):
    try:
        os.makedirs(path)
    except:
        shutil.rmtree(path)
        os.makedirs(path)


def get_prod_dockerfile(contribs):
    return f"""
# =====================
# Autogoal production image
# ---------------------

FROM autogoal/autogoal:core

USER root

RUN ./install-package.sh common remote {contribs}

EXPOSE 8000

COPY ./storage /home/coder/autogoal/autogoal-export/storage

CMD ["python3", "-m", "autogoal", "remote", "serve" ]

"""


class AlgorithmConfig:
    def __init__(self, name, module, args):
        self.name = name
        self.module = module
        self.args = args

    def to_yaml(self, path: Path):
        info = {"name": self.name, "module": self.module, "params": self.args}
        with open(path / "algorithm.yml", "w") as fd:
            yaml.safe_dump(info, fd)

    @classmethod
    def from_yaml(self, path: Path):
        with open(path / "algorithm.yml", "r") as fd:
            values = yaml.safe_load(fd)
            return AlgorithmConfig(
                values.get("name"), values.get("module"), values.get("params")
            )

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        args = ", ".join([f"{a}={b}" for a, b in self.args.items()])
        return f"{self.name} ({args})"


def inspect_storage(path: Path) -> "str":
    main_folder = path / "storage"
    algorithms_path = main_folder / "algorithms"
    general_config = main_folder / "algorithms.yml"

    with open(general_config, "r") as fd:
        pipeline = yaml.safe_load(fd)
        inputs = pipeline["inputs"]
        count = len(pipeline["algorithms"])

    algorithms = [
        AlgorithmConfig.from_yaml(algorithms_path / str(i)) for i in range(count)
    ]

    return str(algorithms) + str(inputs)


def generate_production_dockerfile(path: Path, contribs: List[str]):
    with open(path / "dockerfile", "w") as fd:
        fd.write(get_prod_dockerfile(" ".join(contribs)))


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(
                os.path.join(root, file),
                os.path.relpath(os.path.join(root, file), os.path.join(path, "..")),
            )


def create_zip_file(folder_path, file_name):
    output_filename = file_name
    shutil.make_archive(output_filename, "zip", folder_path)
    return output_filename
