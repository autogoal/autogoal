from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException

from autogoal.contrib import find_classes
from autogoal.utils.remote._algorithm import RemoteAlgorithmDTO, RemoteAttrCallDTO, RemoteInstantiateRequest
from autogoal.utils._dynamic import dynamic_call
import json

app = FastAPI()
algorithms = find_classes()
algorithm_pool = {}


@app.get("/")
async def root():
    return {"message": "Service Running"}

@app.get("/algorithms")
async def get_algorithms(request: Request):
    """
    Returns exposed algorithms
    """
    remote_algorithms = [RemoteAlgorithmDTO.from_algorithm_class(a) for a in algorithms]
    return {
        "message": f"Exposing {str(len(algorithms))} algoritghms: {', '.join([a.__name__ for a in algorithms])}",
        "algorithms": remote_algorithms,
    }


@app.post("/algorithm/call")
async def post_call(request: RemoteAttrCallDTO):
    dto = RemoteAlgorithmDTO.parse_obj(request.algorithm_dto)
    inst = algorithm_pool.get(dto.id)
    if (inst == None):
        raise HTTPException(400, f"Algorithm instance with id={dto.id} not found")
    
    result = dynamic_call(inst, request.attr, *json.loads(request.args), **json.loads(request.kwargs))
    return result


@app.post("/algorithm/instantiate")
async def post_call(request: RemoteInstantiateRequest):
    dto = RemoteAlgorithmDTO.parse_obj(request.algorithm_dto)
    cls = dto.get_original_class()
    inst = cls(*json.loads(request.args), **json.loads(request.kwargs))
    algorithm_pool[dto.id] = inst
    return {
        "message" : "success"
    }

@app.delete("/algorithm")
async def delete_algorithm(request: RemoteAlgorithmDTO):
    algorithm_pool.pop(request.id)
    return {
        "message" : f"deleted instance for {request.id}"
    }



def run(ip=None, port=None):
    """
    Starts HTTP API with specified model.
    """
    uvicorn.run(app, host=ip or "0.0.0.0", port=port or 8000)
