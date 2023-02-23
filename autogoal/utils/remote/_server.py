import inspect
import json
import pickle
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException

from autogoal.contrib import find_classes
from autogoal.utils import RestrictedWorkerByJoin
from autogoal.utils import Gb, Hour, Kb, Mb, Min, Sec
from autogoal.utils._dynamic import dynamic_call
from autogoal.utils.remote._algorithm import (AttrCallRequest,
                                              InstantiateRequest,
                                              RemoteAlgorithmDTO, decode,
                                              dumps, encode, loads)

                                              

app = FastAPI()

# get references for every algorithm in contribs
algorithms = find_classes()

# simple set for pooling algorithm instances. If instances
# are not properly deleted can (and will) fill the memory
algorithm_pool = {}

# sets the RAM usage restriction for remote calls. This will only affect
# remote attribute calls and is ignored during the instance creation. 
# Defaults to 4Gb.
remote_call_ram_limit = 4*Gb

# sets the remote call timeout. This will only affect
# remote attribute calls and is ignored during the instance creation. 
# Defaults to 20 Sec.
remote_call_timeout = 20*Sec


@app.get("/")
async def root():
    return {"message": "Service Running"}


@app.get("/algorithms")
async def get_exposed_algorithms(request: Request):
    """
    Returns exposed algorithms
    """
    remote_algorithms = [RemoteAlgorithmDTO.from_algorithm_class(a) for a in algorithms]
    return {
        "message": f"Exposing {str(len(algorithms))} algoritghms: {', '.join([a.__name__ for a in algorithms])}",
        "algorithms": remote_algorithms,
    }


@app.post("/algorithm/call")
async def post_call(request: AttrCallRequest):
    id = uuid.UUID(request.instance_id, version=4)
    inst = algorithm_pool.get(id)
    if inst == None:
        raise HTTPException(400, f"Algorithm instance with id={id} not found")

    attr = getattr(inst, request.attr)
    is_callable = hasattr(attr, "__call__")

    try:
        result = (
            dynamic_call(inst, request.attr, *loads(request.args), **loads(request.kwargs))
            if is_callable
            else attr
        )
    except Exception as e:
        raise HTTPException(500, str(e))

    return {"result": dumps(result)}


@app.post("/algorithm/has_attr")
async def has_attr(request: AttrCallRequest):
    id = uuid.UUID(request.instance_id, version=4)
    inst = algorithm_pool.get(id)
    if inst == None:
        raise HTTPException(400, f"Algorithm instance with id={id} not found")

    try:
        attr = getattr(inst, request.attr)
        result = True
    except:
        result = False

    return {"exists": result, "is_callable": result and hasattr(attr, "__call__")}


@app.post("/algorithm/instantiate")
async def post_call(request: InstantiateRequest):
    dto = RemoteAlgorithmDTO.parse_obj(request.algorithm_dto)
    cls = dto.get_original_class()
    new_id = uuid.uuid4()
    algorithm_pool[new_id] = cls(*loads(request.args), **loads(request.kwargs))
    return {"message": "success", "id": new_id}


@app.delete("/algorithm/{raw_id}")
async def delete_algorithm(raw_id):
    id = uuid.UUID(raw_id, version=4)

    try:
        algorithm_pool.pop(id)
    except KeyError:
        # do nothing, key is already out of the pool. Dont ask that many questions...
        pass

    return {"message": f"deleted instance with id={id}"}


def run(ip=None, port=None):
    """
    Starts HTTP API with specified model.
    """
    uvicorn.run(app, host=ip or "0.0.0.0", port=port or 8000)


if __name__ == "__main__":
    run()
