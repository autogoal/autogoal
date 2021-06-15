import sparseml
import autogoal
import random as rd
from autogoal.datasets import cars
from autogoal.kb import (MatrixContinuousDense, 
                         Supervised, 
                         VectorCategorical)
from autogoal.ml import AutoML

# Load dataset
X, y = cars.load()

# Instantiate AutoML and define input/output types
automl = AutoML(
    input=(MatrixContinuousDense, 
           Supervised[VectorCategorical]),
    output=VectorCategorical
)

automl.fit(X, y)

def func(i):
    return i-rd.random()



from sparseml.keras.optim import ScheduledModifierManager
manager = ScheduledModifierManager.from_yaml(PATH_TO_RECIPE)
manager.apply(model)



def SparseMLKeras(model,path_to_recipe):
    manager = ScheduledModifierManager.from_yaml(path_to_recipe)
    newmodel=manager.apply(model)
    return newmodel

