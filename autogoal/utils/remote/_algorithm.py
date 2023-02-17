import json
import re
from typing import Dict

import dill as pickle
from pydantic import BaseModel
from requests.api import post

from autogoal.kb import AlgorithmBase
from autogoal.utils._dynamic import dynamic_imp

contrib_pattern = r'autogoal\.contrib\.(?P<contrib>\w+)\.?.*'

def decode_type(type):
    return pickle.dumps(type).decode("latin1")


def encode_type(code: str):
    return code.encode("latin1")


class RemoteAlgorithmDTO(BaseModel):
    name: str
    module: str
    contrib: str
    input_args: str
    init_input_types: str
    input_types: str
    output_type: str

    @staticmethod
    def from_algorithm_class(algorithm_cls):
        name = algorithm_cls.__name__
        module = algorithm_cls.__module__
        contrib = re.search(contrib_pattern, module).group("contrib")
        input_args = decode_type(algorithm_cls.input_args())
        init_input_types = decode_type(algorithm_cls.init_input_types())
        input_types = decode_type(algorithm_cls.input_types())
        output_type = decode_type(algorithm_cls.output_type())
        return RemoteAlgorithmDTO(
            name=name,
            module=module,
            contrib=contrib,
            input_args=input_args,
            init_input_types=init_input_types,
            input_types=input_types,
            output_type=output_type,
        )

    def get_id(self):
        return self.module + self.name

    def get_original_class(self):
        return dynamic_imp(self.module, self.name)

    @staticmethod
    def get_original_class_from_dict(data: Dict):
        dto = RemoteAlgorithmDTO.parse_obj(data)
        return dto.get_original_class()

    def build_algorithm_class(self, ip: str = None, port: int = None):
        ip = ip or "0.0.0.0"
        port = port or 8000
        id = f"{ip}:{port}-{self.name}"

        cls = type(id, (RemoteAlgorithmBase,), {})
        cls.input_types = lambda: pickle.loads(encode_type(self.input_types))
        cls.output_type = lambda: pickle.loads(encode_type(self.output_type))
        cls.dto = self
        cls.contrib = self.contrib
        cls.ip = ip
        cls.port = port

        return cls

class RemoteAlgorithmBase(AlgorithmBase):
    contrib = "remote"
    dto: RemoteAlgorithmDTO = None
    ip: str = None
    port: int = None

    def __new__(cls: type, *args, **kwargs):
        call = RemoteInstantiateRequest.build(cls.dto, args, kwargs)
        response = post(
            f"http://{cls.ip or '0.0.0.0'}:{cls.port or 8000}/algorithm/instantiate",
            json=call.dict(),
        )
        if response.status_code == 200:
            return super().__new__(cls)

        raise Exception(response.reason)

    def __getattribute__(self, name):
        attr = object.__getattribute__(self, name)
        if hasattr(attr, "__call__") and attr.__name__ != "proxy_call":

            def intercept(*args, **kwargs):
                return self.proxy_call(attr, *args, **kwargs)

            return intercept
        else:
            return attr

    def proxy_call(self, attr, *args, **kwargs):
        call = RemoteAttrCallDTO.build(self.dto, attr.__name__, args, kwargs)
        response = post(
            f"http://{self.ip or '0.0.0.0'}:{self.port or 8000}/algorithm/call",
            json=call.dict(),
        )
        if response.status_code == 200:
            return response.content

    def run(self, *args):
        pass


class RemoteAttrCallDTO(BaseModel):
    attr: str
    args: str
    kwargs: str
    algorithm_dto: Dict

    @staticmethod
    def build(dto: RemoteAlgorithmDTO, attr: str, args, kwargs):
        return RemoteAttrCallDTO(
            attr=attr,
            args=json.dumps(args),
            kwargs=json.dumps(kwargs),
            algorithm_dto=dto,
        )


class RemoteInstantiateRequest(BaseModel):
    args: str
    kwargs: str
    algorithm_dto: Dict

    @staticmethod
    def build(dto: RemoteAlgorithmDTO, args, kwargs):
        return RemoteInstantiateRequest(
            args=json.dumps(args),
            kwargs=json.dumps(kwargs),
            algorithm_dto=dto,
        )