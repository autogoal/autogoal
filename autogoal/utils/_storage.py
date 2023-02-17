import yaml
from pathlib import Path

dockerfile = """
# =====================
# Autogoal base image
# ---------------------

FROM autogoal:base

EXPOSE 8000

USER root

COPY ./storage /home/coder/autogoal/storage

SHELL ["conda", "run", "-n", "autogoal", "/bin/bash", "-c"]

RUN chmod +x ./storage/contribs.sh && ./storage/contribs.sh

CMD [ "conda", "run", "-n", "autogoal", "python3", "-m", "autogoal", "ml", "serve" ]

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


def generate_production_dockerfile(path: Path):
    with open(path / "dockerfile", "w") as fd:
        fd.write(dockerfile)
