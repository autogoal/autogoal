from typing import Any
from fastapi import FastAPI
from pathlib import Path
from autogoal.ml import AutoML
from pydantic import BaseModel
import uvicorn


class Body(BaseModel):
    values: list

app = FastAPI()


@app.get("/")
async def root():
    model = AutoML.folder_load(Path('.'))

    return {"message": str(model.best_pipeline_.input_types)}

@app.post("/")
async def postroot(body: Body):
    model = AutoML.folder_load(Path('.'))

    print(body)

    return {"message": str(model.predict(body.values))}

    ##return {"message": str(model.best_pipeline_.input_types)}

def run():
    uvicorn.run(app, host="0.0.0.0", port=8000)