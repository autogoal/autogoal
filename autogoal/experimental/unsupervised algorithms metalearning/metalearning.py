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


# Returns the amount of data attributes.
@feature_extractor
def attributes_amount(X, y=None):
    return len(X[0])


# Returns log_2 of the amount of data attributes.
@feature_extractor
def attributes_amount_log_2(X, y=None):
    return math.log2(len(X[0]))


# Returns log_10 of the amount of data attributes.
@feature_extractor
def attributes_amount_log_10(X, y=None):
    return math.log10(len(X[0]))


# Returns the amount of continuous attributes.
@feature_extractor
def continuous_amount(X, y=None):
    amount = 0
    for val in X[0]:
        if isinstance(val, (float))
            amount += 1
    return amount

# Returns the proportion of continuous attributes.
@feature_extractor
def continuous_proportion(X, y=None):
    amount = 0
    for val in X[0]:
        if isinstance(val, (float))
            amount += 1
    return (float)amount / (float)len(X[0])


# Returns the mean absolute correlation between continuous attributes.
@feature_extractor
def continuous_mean_absolute_correlation(X, y=None):
    continuous_attributes = []
    for i in range(0, len(X[0])):
        if isinstance(X[0][i], (float))
            continuous_attributes.append(i)
    
    means = []
    variances = []
    for i in range(0, len(continuous_attributes)):
        mean = 0
        squares_sum = 0
        for j in range(0, len(X)):
            squares_sum += X[j][continuous_attributes[i]] ** 2
            mean += X[j][continuous_attributes[i]]
        mean /= len(X)
        means.append(mean)
        variances.append(squares_sum / len(X) - mean ** 2)
    
    correlations = []

    for i in range(0, len(continuous_attributes - 1)):
        for j in range(i + 1, len(continuous_attributes)):
            product_sum = 0
            for k in range(0, len(X)):
                product_sum += X[k][continuous_attributes[i]] * X[k][continuous_attributes[j]]
            covariance = product_sum / len(X) - means[i] * means[j]
            correlations.append(covariance / (variances[i] * variances[j]) ** 0.5)
    
    correlation_sum = 0
    for i in range(0, len(correlations)):
        correlation_sum += correlations[i]
    
    return correlation_sum / len(correlations)


# Returns the mean skewness of continuous attributes.
@feature_extractor
def continuous_mean_skewness(X, y=None):
    continuous_attributes = []
    for i in range(0, len(X[0])):
        if isinstance(X[0][i], (float))
            continuous_attributes.append(i)
    
    means = []
    variances = []
    standard_desviations = []
    for i in range(0, len(continuous_attributes)):
        mean = 0
        squares_sum = 0
        for j in range(0, len(X)):
            squares_sum += X[j][continuous_attributes[i]] ** 2
            mean += X[j][continuous_attributes[i]]
        mean /= len(X)
        means.append(mean)
        variances.append(squares_sum / len(X) - mean ** 2)
        standard_desviations.append(variances[len(variances) - 1] ** 0.5)


    skewness = []
    skewness_sum = 0
    n = len(X)

    for i in range(0, len(continuous_attributes)):
        S_above = 0
        S_below = 0
        for j in range(0, n):
            mirror_val = (X[j][continuous_attributes[i]] - means[i]) ** 3
            if mirror_val > 0:
                S_above += mirror_val
            else:
                S_above += -mirror_val
        skewness.append((n / (standard_desviations[i] ** 3 * (n - 1) * (n - 2))) * (S_above - S_below))
        skewness_sum += skewness[len(skewness) - 1]

    return skewness_sum / len(continuous_attributes)


# Returns the mean kurtosis of continuous attributes.
@feature_extractor
def continuous_mean_kurtosis(X, y=None):
    continuous_attributes = []
    for i in range(0, len(X[0])):
        if isinstance(X[0][i], (float))
            continuous_attributes.append(i)
    
    means = []
    variances = []
    standard_desviations = []
    for i in range(0, len(continuous_attributes)):
        mean = 0
        squares_sum = 0
        for j in range(0, len(X)):
            squares_sum += X[j][continuous_attributes[i]] ** 2
            mean += X[j][continuous_attributes[i]]
        mean /= len(X)
        means.append(mean)
        variances.append(squares_sum / len(X) - mean ** 2)
        standard_desviations.append(variances[len(variances) - 1] ** 0.5)
    
    kurtosis = []
    kurtosis_sum = 0
    n = len(X)
    left_coeficient = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
    right_substractor = (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))

    for i in range(0, len(continuous_attributes)):
        central_value = 0
        for j in range(0, n):
            central_value += ((X[j][continuous_attributes[i]] - means[i]) ** 4) / (standard_desviations[i] ** 4)
        kurtosis.append(left_coeficient * central_value - right_substractor)
        kurtosis_sum += kurtosis[len(kurtosis) - 1]
    
    return kurtosis_sum / len(continuous_attributes)
    
