from typing import Any
from fastapi import FastAPI, Response, Request
from pathlib import Path
from pydantic import BaseModel
from autogoal.contrib import find_classes
import uvicorn

app = FastAPI()
algorithms = find_classes()

@app.get("/")
async def root():
    return {"message": "Service Running"}

@app.get("/algorithms")
async def input(request: Request):
    """
    Returns exposed algorithms
    """
    return {"message": "Exposing " + str(len(algorithms)) + " algoritghms: " + ', '.join([a.__name__ for a in algorithms])}

def run(ip=None, port=None):
    """
    Starts HTTP API with specified model.
    """
    uvicorn.run(app, host=ip or "0.0.0.0", port=port or 8000)