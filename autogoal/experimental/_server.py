from typing import Any
from fastapi import FastAPI, Response, Request
from pathlib import Path
from autogoal.ml import AutoML
from pydantic import BaseModel
from autogoal.utils import inspect_storage
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

def run(model = None):
    app.model = model or AutoML.folder_load(Path('.'))
    uvicorn.run(app, host="0.0.0.0", port=8000) 