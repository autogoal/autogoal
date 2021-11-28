from typing import Any
from fastapi import FastAPI, Response
from pathlib import Path
from autogoal.ml import AutoML
from pydantic import BaseModel
from autogoal.utils import inspect_storage
import uvicorn


class Body(BaseModel):
    values: Any

app = FastAPI()

model = AutoML.folder_load(Path('.'))

@app.get("/input")
async def root():
    """
    Returns the model input type
    """
    return {"message": str(model.best_pipeline_.input_types[0])}

@app.get("/output")
async def output():
    """
    Returns the model output type
    """
    return {"message": str(model.best_pipeline_.algorithms[-1].__class__.output_type())}

@app.get("/inspect")
async def root():
    """
    Returns the model inspect command
    """
    return {"message": str(inspect_storage(Path('.')))}

@app.post("/")
async def eval(t: Body):
    """
    Returns the model prediction over the provided values
    """
    input_type = model.best_pipeline_.input_types[0]

    output_type = model.best_pipeline_.algorithms[-1].__class__.output_type()

    data = input_type.from_json(t.values)

    result = model.predict(data)

    return Response(content=output_type
        .to_json(result), media_type="application/json")

def run():
    uvicorn.run(app, host="0.0.0.0", port=8000)