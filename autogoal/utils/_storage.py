import yaml
from pathlib import Path

class AlgorithmConfig:
    
    def __init__(self, name, module, args):
        self.name = name
        self.module = module
        self.args = args

    def to_yaml(self, path: Path):
        info = { 
            "name": self.name,
            "module": self.module,
            "params": self.args
            }
        with open(path/ "algorithm.yml", "w") as fd:
            yaml.safe_dump(info, fd)
    
    @classmethod 
    def from_yaml(self, path: Path):
        with open(path/ "algorithm.yml", "r") as fd:
            values = yaml.safe_load(fd)
            return AlgorithmConfig(values.get('name'), values.get('module'), values.get('params'))

    def __repr__(self)-> str:
        return self.__str__()
    
    def __str__(self) -> str:
        print(self.args)
        args = ", ".join([f"{a}={b}" for a,b in self.args.items()])
        return f"{self.name} ({args})"


def inspect_storage(path : Path) -> "str":
    main_folder = path / 'storage'
    algorithms_path = main_folder / 'algorithms'
    general_config = main_folder / 'algorithms.yml'

    with open(general_config, "r") as fd:
        inputs = yaml.safe_load(fd)["inputs"]

    with open(general_config, "r") as fd:
        pipeline = yaml.safe_load(fd)
        count = len(pipeline)

    algorithms = [AlgorithmConfig.from_yaml(algorithms_path / str(i)) for i in range(count)]

    return str(algorithms) + str(inputs)

