import json
import re
from typing import Dict

import dill
import pickle
import uuid
from pydantic import BaseModel
from requests.api import post, delete

from autogoal.kb import AlgorithmBase
from autogoal.utils._dynamic import dynamic_imp

contrib_pattern = r"autogoal\.contrib\.(?P<contrib>\w+)\.?.*"


def dumps(data: object, use_dill=False) -> str:
    data = dill.dumps(data) if use_dill else pickle.dumps(data)
    return decode(data)


def decode(data: bytes, use_dill=False) -> str:
    return data.decode("latin1")


def loads(data: str, use_dill=False):
    raw_data = data.encode("latin1")
    return dill.loads(raw_data) if use_dill else pickle.loads(raw_data)


def encode(code: str):
    return code.encode("latin1")


class RemoteAlgorithmDTO(BaseModel):
    name: str
    module: str
    contrib: str
    input_args: str
    init_input_types: str
    inner_signature: str
    input_types: str
    output_type: str

    @staticmethod
    def from_algorithm_class(algorithm_cls):
        name = algorithm_cls.__name__
        module = algorithm_cls.__module__
        contrib = re.search(contrib_pattern, module).group("contrib")
        input_args = dumps(algorithm_cls.input_args())
        init_input_types = dumps(algorithm_cls.init_input_types(), use_dill=True)
        inner_signature = dumps(algorithm_cls.get_inner_signature(), use_dill=True)
        input_types = dumps(algorithm_cls.input_types())
        output_type = dumps(algorithm_cls.output_type())

        return RemoteAlgorithmDTO(
            name=name,
            module=module,
            contrib=contrib,
            input_args=input_args,
            init_input_types=init_input_types,
            inner_signature=inner_signature,
            input_types=input_types,
            output_type=output_type,
        )

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
        cls.input_types = lambda: loads(self.input_types)
        cls.init_input_types = lambda: loads(self.init_input_types, use_dill=True)
        cls.get_inner_signature = lambda: loads(self.inner_signature, use_dill=True)
        cls.output_type = lambda: loads(self.output_type)
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
        call = InstantiateRequest.build(cls.dto, args, kwargs)
        response = post(
            f"http://{cls.ip or '0.0.0.0'}:{cls.port or 8000}/algorithm/instantiate",
            json=call.dict(),
        )

        if response.status_code == 200:
            content = json.loads(response.content)
            instance = super().__new__(cls)
            instance.id = uuid.UUID(content["id"], version=4)
            return instance

        raise Exception(response.reason)

    def __del__(self):
        delete(
            f"http://{self.ip or '0.0.0.0'}:{self.port or 8000}/algorithm/{self.id}",
        )

    def __getattribute__(self, name):
        # wrapper function for proxy calls to be returned
        def intercept(*args, **kwargs):
            return self.proxy_call(name, *args, **kwargs)
        
        # first check if the attr is present in this class definition and ignore calls 
        # to proxy_call itself as any call to intercept() would trigger
        try:
            attr = object.__getattribute__(self, name)
            if hasattr(attr, "__call__") and attr.__name__ != "proxy_call":
                return intercept
        except:
            # if the attr is not present in this class then just send the intercept
            return intercept

    def proxy_call(self, attr_name, *args, **kwargs):
        call = AttrCallRequest.build(str(self.id), attr_name, args, kwargs)
        response = post(
            f"http://{self.ip or '0.0.0.0'}:{self.port or 8000}/algorithm/call",
            json=call.dict(),
        )
        if response.status_code == 200:
            content = json.loads(response.content)
            return loads(content["result"])

    def run(self, *args):
        pass


class AttrCallRequest(BaseModel):
    instance_id: str
    attr: str
    args: str
    kwargs: str

    @staticmethod
    def build(instance_id: str, attr: str, args, kwargs):
        return AttrCallRequest(
            instance_id=instance_id,
            attr=attr,
            args=dumps(args),
            kwargs=dumps(kwargs),
        )


class InstantiateRequest(BaseModel):
    args: str
    kwargs: str
    algorithm_dto: Dict

    @staticmethod
    def build(dto: RemoteAlgorithmDTO, args, kwargs):
        return InstantiateRequest(
            args=dumps(args),
            kwargs=dumps(kwargs),
            algorithm_dto=dto,
        )
