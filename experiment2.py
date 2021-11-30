from enum import auto
from numpy.lib.function_base import diff
from autogoal.datasets import cars
from autogoal.kb import MatrixContinuousDense, Supervised, VectorCategorical, Categorical, Seq, Word, Label, Sentence, Tensor, Dense
from autogoal.kb import build_pipeline_graph, SemanticType, Pipeline
from autogoal.ml import AutoML
from autogoal.contrib import find_classes
from time import sleep, time
from pathlib import Path
import os
import subprocess
from json import JSONEncoder, dumps
from numpy import ndarray
import numpy
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.model_selection import train_test_split
from pandas import DataFrame
import shutil
from autogoal.experimental import run
from fastapi.testclient import TestClient
import threading

class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

def inference_timing(dataframe: dict, automl: AutoML, x) -> None:
    try:
        startime = time()

        y = automl.predict(x)

        endtime = time()

        print(y)

        difference = endtime - startime

        dataframe["inference"].append(difference)
    except:
        dataframe["inference"].append(0)  

def api_test(dataframe: dict, x):
    import requests

    p = subprocess.Popen(["python3", "-m", "autogoal", "ml", "serve"], stdin=None, stdout=None, stderr=None)

    try:

        inside = dumps(x, cls=NumpyEncoder)

        data = {"values":inside}

        sleep(10)

        startime = time()

        response = requests.post('http://localhost:8000', json=data)

        endtime = time()

        print(response)

        difference = endtime - startime

        dataframe["api"].append(difference)

        
    except subprocess.SubprocessError as e: 
        pass
    except:
        dataframe["api"].append(0)

    p.terminate()


dataframe = {
    t:[] for t in ["inference", "api", "pipeline"]
}

for i in range(1000):
    X, y = make_classification()
    #X, y = make_regression()

    print(i)

    _, x, _, _ = train_test_split(X, y, test_size=0.1)

    automl = AutoML(
        input=(MatrixContinuousDense, Supervised[VectorCategorical]),
        output=VectorCategorical,
        search_iterations=1
    )

    # Run the pipeline search process
    automl.fit(X, y)

    automl.folder_save(Path('.'))

    inference_timing(dataframe, automl, x)

    api_test(dataframe, x)

    dataframe["pipeline"].append(str(automl.best_pipeline_))

_dataframe = DataFrame(dataframe)

print(_dataframe)

_dataframe.to_csv('experiment2.csv')