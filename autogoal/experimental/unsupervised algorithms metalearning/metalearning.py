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
            if(not( X[j][i] is None) or X[j][i] == True or X[j][i] == False or X[j][i] == 0 or X[j][i] == 1):
                pass
            else:
                binary = False
                break
        if(binary):
           count+=1 
    return count

# Returns the proportion of binary attributes.
@feature_extractor
def binary_proportion(X, y=None):
    count = 0
    for i in range(0, len(X[0])):
        binary = True
        for j in range(0, len(X)):
            if(not( X[j][i] is None) or X[j][i] == True or X[j][i] == False or X[j][i] == 0 or X[j][i] == 1):
                pass
            else:
                binary = False
                break
        if(binary):
           count+=1 
    return count/len(X[0])

# Returns the amount of discrete attributes.
@feature_extractor
def discrete_amount(X, y=None):
    count = 0
    for i in range(0, len(X[0])):
        discrete = True
        for j in range(0, len(X)):
            if(not(X[j][i] is None) and not isinstance(X[j][i], (int))):
                pass
            else:
                discrete = False
                break
        if(discrete):
           count+=1 
    return count

# Returns the proportion of discrete attributes.
@feature_extractor
def discrete_proportion(X, y=None):
    count = 0
    for i in range(0, len(X[0])):
        discrete = True
        for j in range(0, len(X)):
            if(not(X[j][i] is None) and not isinstance(X[j][i], (int))):
                pass
            else:
                discrete = False
                break
        if(discrete):
           count+=1 
    return count/len(X[0])

# Returns the amount of continuos attributes.
def continuos_amount(X, y=None):
    count = 0
    for i in range(0, len(X[0])):
        continuos = True
        for j in range(0, len(X)):
            if(not(X[j][i] is None) and not isinstance(X[j][i], (float))):
                pass
            else:
                continuos = False
                break
        if(continuos):
           count+=1 
    return count

# Returns the proportion of continuos attributes.
@feature_extractor
def discrete_proportion(X, y=None):
    count = 0
    for i in range(0, len(X[0])):
        continuos = True
        for j in range(0, len(X)):
            if(not(X[j][i] is None) and not isinstance(X[j][i], (float))):
                pass
            else:
                continuos = False
                break
        if(continuos):
           count+=1 
    return count/len(X[0])

# Returns the ratio of the number of examples by the number of attributes. An indicator of 
# the number of examples available to the number of attributes.
@feature_extractor
def examples_by_attributes_ratio(X, y=None):
    attributesCount = len(X[0])
    examples = attributesCount * len(X)
    return examples/attributesCount

# Percentage of missing values. An indicator of the quality of the data.
@feature_extractor
def missing_values_percentage(X, y=None):
    count = 0
    for i in range(0, len(X[0])):
        for j in range(0, len(X)):
            if(X[j][i] is None):
                count+=1 
    return (count/(len(X[0])*len(X)))*100

# Instances Amount. An indicator of the quality of the data.
@feature_extractor
def instances_amount(X, y=None):
    count = 0
    for i in range(0, len(X[0])):
        for j in range(0, len(X)):
            if(not(X[j][i] is None)):
                count+=1 
    return count

# Percentage of attributes kept after the application of the attribute selection filter.
@feature_extractor
def kept_attributes_percentage(X, X_before, y=None):
    attributesBefore = len(X_before[0])
    attributesAfter = len(X[0])
    return (attributesAfter/attributesBefore)*100

# Returns the mean entropy of discrete attributes.
@feature_extractor
def discrete_mean_entropy(X, y=None):
    discreteAttributes = []
    for i in range(0, len(X[0])):
        discrete = True
        for j in range(0, len(X)):
            if(not(X[j][i] is None) and not isinstance(X[j][i], (int))):
                pass
            else:
                discrete = False
                break
        if(discrete):
            discreteAttributes.append(i)
    entropy = []
    for attributeColumn in discreteAttributes:
        entropyPi = {}
        for j  in range(0, len(X)):
            if (entropyPi.__contains__(X[j][attributeColumn])):
                entropyPi[X[j][attributeColumn]] += 1
            else:
                entropyPi[X[j][attributeColumn]] = 1
        entropyValue = 0
        for key in entropyPi:
            entropyValue += (entropyPi[key]*math.log2(entropyPi[key]))
        entropy.append(entropyValue)
    entropyValue = 0
    for value in entropy:
        entropyValue += value
    return entropyValue/len(entropy)
