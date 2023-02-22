import json
import re
from typing import Dict

import dill
import pickle
import uuid
from pydantic import BaseModel
from requests.api import post, delete
from functools import partial
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

    def _proxy_call(self, attr_name, *args, **kwargs):
        print(f"calling get attr {attr_name}")
        call = AttrCallRequest.build(str(self.id), attr_name, args, kwargs)
        response = post(
            f"http://{self.ip or '0.0.0.0'}:{self.port or 8000}/algorithm/call",
            json=call.dict(),
        )
        if response.status_code == 200:
            content = json.loads(response.content)
            return loads(content["result"])

    def _has_attr(self, attr_name):
        print(f"calling has attr {attr_name}")
        call = AttrCallRequest.build(str(self.id), attr_name, None, None)
        response = post(
            f"http://{self.ip or '0.0.0.0'}:{self.port or 8000}/algorithm/has_attr",
            json=call.dict(),
        )
        if response.status_code == 200:
            return RemoteAttrInfo.construct(**json.loads(response.content))

    def __getattribute__(self, name):
        # Calls to proxy_call are not supposed to be proxied.
        # Check for attributes from the local instance
        if (
            name == "_proxy_call"
            or name == "_has_attr"
            or name == "id"
            or name == "ip"
            or name == "port"
        ):
            return object.__getattribute__(self, name)

        # get remote information for the attribute.
        # do nothing if the attribute does not exists in the remote instance and return None.
        remote_attr_info = self._has_attr(name)
        if remote_attr_info.exists:
            if remote_attr_info.is_callable:
                # if attribute is callable then return a partial function based on _proxy_call
                #  which will take up the args and kwargs specified by the caller
                return partial(self._proxy_call, name)

            # if object is not callable then return the exact attr from the remote object
            return self._proxy_call("__getattribute__", name)

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


class RemoteAttrInfo(BaseModel):
    attr: str
    exists: bool
    is_callable: bool
