from typing import List
import yaml
from inspect import getsourcefile
from os.path import abspath, dirname, join
from yamlable import yaml_info, YamlAble

config_dir = dirname(abspath(getsourcefile(lambda:0)))

@yaml_info(yaml_tag_ns='autogoal.remote.connectionAlias')
class Alias(YamlAble):
    def __init__(self, name, ip, port):
        self.name = name
        self.ip = ip
        self.port = port

@yaml_info(yaml_tag_ns='autogoal.remote.connectionConfig')
class ConnectionConfig(YamlAble):
    def __init__(self, connections):
        self.connections = connections

def load_config() -> ConnectionConfig:
    path = join(config_dir, "connections.yml")
    result = None
    try:
        with open(path, "r") as fd:
            result = yaml.safe_load(fd)
    except IOError as e:
        config = ConnectionConfig([])
        with open(path, "w") as fd:
            yaml.dump(config, fd)
            result = config
    return result

def save_config(config: ConnectionConfig):
    path = join(config_dir, "connections.yml")
    with open(path, "w") as fd:
        yaml.dump(config, fd)

def store_connection(ip: str, port: int, alias: str):
    config = load_config()
    calias = Alias(alias, ip, port)
    config.connections.append(calias)
    save_config(config)


def clear_connetions():
    config = load_config()
    config.connections = []
    save_config(config)