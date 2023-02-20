import json
import pickle
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException

from autogoal.contrib import find_classes
from autogoal.utils._dynamic import dynamic_call
from autogoal.utils.remote._algorithm import (
    AttrCallRequest,
    InstantiateRequest,
    RemoteAlgorithmDTO,
    decode,
    dumps,
    encode,
    loads,
)

app = FastAPI()
algorithms = find_classes()
algorithm_pool = {}


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

    return {
        "result": dumps(
            dynamic_call(
                inst, request.attr, *loads(request.args), **loads(request.kwargs)
            )
        )
    }


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
    algorithm_pool.pop(id)
    return {"message": f"deleted instance with id={id}"}


def run(ip=None, port=None):
    """
    Starts HTTP API with specified model.
    """
    uvicorn.run(app, host=ip or "0.0.0.0", port=port or 8000)


if __name__ == "__main__":
    run()
