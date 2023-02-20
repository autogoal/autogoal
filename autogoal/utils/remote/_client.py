import json
from typing import Any, Dict, List, Set, Tuple, Type

from requests.api import get

from autogoal.utils.remote import RemoteAlgorithmBase, RemoteAlgorithmDTO
from autogoal.utils.remote.config import load_config


def get_algorithms(
    ip: str = None, port: int = None, alias: str = None
) -> List[RemoteAlgorithmBase]:
    """Gets valid algorithms from remote AutoGOAL instances. If `alias` is specified and
    a connection alias with that name is already stored then `ip` and `port` are retrieved from configuration, hence ignoring the arguments values.
    """
    if alias is not None:
        config = load_config()
        c_alias = config.connections.get(alias)
        if c_alias is not None:
            ip = c_alias.ip
            port = c_alias.port

    response = get(f"http://{ip or '0.0.0.0'}:{port or 8000}/algorithms")
    raw_algorithms = json.loads(response.content)["algorithms"]
    algorithms = [
        RemoteAlgorithmDTO(**ralg).build_algorithm_class(ip, port)
        for ralg in raw_algorithms
    ]
    return algorithms
