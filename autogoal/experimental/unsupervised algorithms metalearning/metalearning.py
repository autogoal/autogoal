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

# Returns log_10 of the amount of data examples. A raw indication of the available amount of data.
@feature_extractor
def examples_amount_log_10(X, y=None):
    return math.log10(len(X))

# Returns the amount of binary attributes.
@feature_extractor
def binary_amount(X, y=None):
    count = 0
    for i in range(0, len(X[0])):
        binary = True
        for j in range(0, len(X)):
            if(X[i][j] == True or X[i][j] == False or X[i][j] == 0 or X[i][j] == 1):
                pass
            else:
                binary = False
                break
        if(binary):
           count+=1 
    return count
