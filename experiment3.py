from enum import auto
from numpy.lib.function_base import diff
from autogoal.datasets import cars
from autogoal.kb import MatrixContinuousDense, Supervised, VectorCategorical, Categorical
from autogoal.ml import AutoML
from time import sleep, time
from pathlib import Path
import os
from json import JSONEncoder, dumps
from numpy import ndarray
import numpy
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.model_selection import train_test_split
from pandas import DataFrame

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

def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size

def storage_size(dataframedict) -> None:
    try:
        dataframe["sizeondisk"].append(get_size("storage"))
    except:
        dataframe["sizeondisk"].append(0)

def docker_test(dataframe: dict, x):
    import requests

    try:
        os.system('docker build --rm --file ./dockerfiles/production/dockerfile -t autogoal:production .')
        os.system(f'docker save -o prod.tar autogoal:production')

        os.system('docker run --rm -p 0.0.0.0:8000:8000/tcp autogoal:production &')

        sleep(5)

        inside = dumps(x, cls=NumpyEncoder)

        data = {"values":inside}


        response = requests.post('http://localhost:8000', json=data)

        os.system('docker kill $(docker ps -q)')
        os.system('docker rm $(docker ps -a -q)')

        os.system(f'docker rmi -f autogoal:production')

        size = os.path.getsize("prod.tar")

        os.remove("prod.tar")

        dataframe["docker"].append(size)

    except:
        dataframe["docker"].append(0)

dataframe = {
    t:[] for t in ["sizeondisk", "docker", "pipeline"]
}

# Load dataset
X, y = cars.load()

for i in range(100):

    print(i)

    X, y = make_classification()
    #X, y = make_regression()

    _, x, _, _ = train_test_split(X, y, test_size=0.1)

    automl = AutoML(
        input=(MatrixContinuousDense, Supervised[VectorCategorical]),
        output=VectorCategorical,
        search_iterations=1
    )

    # Run the pipeline search process
    automl.fit(X, y)

    storage_size(dataframe)

    docker_test(dataframe, x)

    sleep(5)

    dataframe["pipeline"].append(str(automl.best_pipeline_))

print(dataframe)

_dataframe = DataFrame(dataframe)

_dataframe.to_csv("experiment3.csv")