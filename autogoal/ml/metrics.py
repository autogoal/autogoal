import inspect
import numpy as np

from autogoal.kb import CategoricalVector
from autogoal.kb import conforms


METRICS = []


def register_metric(func):
    METRICS.append(func)
    return func


def find_metric(*types):
    for metric_func in METRICS:
        signature = inspect.signature(metric_func)

        if len(types) != len(signature.parameters):
            break

        for type_if, type_an in zip(types, signature.parameters):
            if not conforms(type_an.annotation, type_if):
                break

        return metric_func

    raise ValueError("No metric found for types: %r" % types)


def accuracy(ytrue: CategoricalVector, ypred: CategoricalVector) -> float:
    return np.mean([1 if yt == yp else 0 for yt,yp in zip(ytrue, ypred)])
