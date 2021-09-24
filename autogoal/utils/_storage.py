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
            yaml.dump(info, fd)
    
    @classmethod 
    def from_yaml(self, path: Path):
        with open(path/ "algorithm.yml", "r") as fd:
            return yaml.load(fd)
