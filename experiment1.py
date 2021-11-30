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
from json import JSONEncoder, dumps
from numpy import ndarray
import numpy
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.model_selection import train_test_split
from pandas import DataFrame
import shutil

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

def save_timing(dataframe: dict, automl: AutoML) -> None:
    try:
        startime = time()

        automl.folder_save(Path('.'))

        endtime = time()

        difference = endtime - startime

        dataframe['save'].append(difference)
    
    except Exception as e:
        print(e)
        dataframe['save'].append(0)

def load_timing(dataframe: dict, previous: AutoML) -> "AutoML":
    try:
        startime = time()

        automl = AutoML.folder_load(Path('.'))

        endtime = time()

        difference = endtime - startime

        dataframe['load'].append(difference)

        return automl
    
    except Exception as e:
        dataframe['load'].append(0)
        print(e)
        return previous

dataframe = {
    t:[] for t in ["save", "load", "pipeline"]
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

    save_timing(dataframe, automl)

    automl = load_timing(dataframe, automl)

    dataframe["pipeline"].append(str(automl.best_pipeline_))

_dataframe = DataFrame(dataframe)

print(_dataframe)

_dataframe.to_csv('experiment1.csv')