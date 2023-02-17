import json
from typing import Any, Dict, List, Set, Tuple, Type

from requests.api import get

from autogoal.utils.remote._algorithm import (RemoteAlgorithmBase,
                                              RemoteAlgorithmDTO)


def get_algotihms(ip: str = None, port: int = None) -> List[RemoteAlgorithmBase]:
    response = get(f"http://{ip or '0.0.0.0'}:{port or 8000}/algorithms")
    raw_algorithms = json.loads(response.content)["algorithms"]
    algorithms = [
        RemoteAlgorithmDTO(**ralg).build_algorithm_class(ip, port)
        for ralg in raw_algorithms
    ]
    return algorithms

# a = get_algotihms("172.20.0.2", 8000)[0]