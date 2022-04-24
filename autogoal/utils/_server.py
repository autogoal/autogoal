from typing import Any
from fastapi import FastAPI, Response, Request
from pathlib import Path
from pydantic import BaseModel
from ._storage import inspect_storage
import uvicorn


class Body(BaseModel):
    values: Any

app = FastAPI()

@app.get("/input")
async def input(request: Request):
    """
    Returns the model input type
    """
    return {"message": str(request.app.model.best_pipeline_.input_types[0])}

@app.get("/output")
async def output(request: Request):
    """
    Returns the model output type
    """
    return {"message": str(request.app.model.best_pipeline_.algorithms[-1].__class__.output_type())}

@app.get("/inspect")
async def inspect():
    """
    Returns the model inspect command
    """
    return {"message": str(inspect_storage(Path('.')))}

@app.post("/")
async def eval(t: Body, request: Request):
    """
    Returns the model prediction over the provided values
    """
    model = request.app.model

    input_type = model.best_pipeline_.input_types[0]

    output_type = model.best_pipeline_.algorithms[-1].__class__.output_type()

    data = input_type.from_json(t.values)

    result = model.predict(data)

    return Response(content=output_type
        .to_json(result), media_type="application/json")

def run(model, ip = None, port = None):
    '''
    Starts HTTP API with specified model. 
    '''
    app.model = model
    uvicorn.run(app, host = ip or "0.0.0.0", port = port or 8000) 

    # def run(model = None, model_path = None, ip = None, port = None):
    # '''
    # Starts HTTP API with specified model and path.
    # '''
    # app.model = model or AutoML.folder_load(Path(model_path or '.'))
    # uvicorn.run(app, host= ip or "0.0.0.0", port= port or 8000) 