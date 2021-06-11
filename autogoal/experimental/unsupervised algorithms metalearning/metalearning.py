import functools
import math

_EXTRACTORS = []

def feature_extractor(func):
    @functools.wraps(func)
    def wrapper(X, y=None):
        try:
            result = func(X, y)
        except:
            result = None
            # cannot apply feature extractor to that dataset

        return {func.__name__: result}

    _EXTRACTORS.append(wrapper)
    return wrapper


# Feature extractor methods


# Returns the amount of attributes.
@feature_extractor
def dimensions(X, y=None):
    return len(X[0])

# Returns the amount of data examples. A raw indication of the available amount of data.
@feature_extractor
def examples_amount(X, y=None):
    return len(X)

# Returns log_2 of the amount of data examples. A raw indication of the available amount of data.
@feature_extractor
def examples_amount_log_2(X, y=None):
    return math.log2(len(X))