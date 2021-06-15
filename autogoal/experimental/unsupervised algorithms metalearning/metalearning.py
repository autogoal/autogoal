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

###################################### SINGLE VALUED META-FEATURES ########################################

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


# Returns the mean absolute correlation (Pearson coefficient) between continuous attributes.
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

# Returns the number of attribute pairs with high correlation (Pearson coefficient)
@feature_extractor
def number_of_high_correlated_pairs(X, y=None):
    numeric_attributes = []
    for i in range(0, len(X[0])):
        if isinstance(X[0][i], (int, float))
            numeric_attributes.append(i)
    
    means = []
    variances = []
    for i in range(0, len(numeric_attributes)):
        mean = 0
        squares_sum = 0
        for j in range(0, len(X)):
            squares_sum += X[j][numeric_attributes[i]] ** 2
            mean += X[j][numeric_attributes[i]]
        mean /= len(X)
        means.append(mean)
        variances.append(squares_sum / len(X) - mean ** 2)
    
    correlations = []
    for i in range(0, len(numeric_attributes - 1)):
        for j in range(i + 1, len(numeric_attributes)):
            product_sum = 0
            for k in range(0, len(X)):
                product_sum += X[k][numeric_attributes[i]] * X[k][numeric_attributes[j]]
            covariance = product_sum / len(X) - means[i] * means[j]
            correlations.append(covariance / (variances[i] * variances[j]) ** 0.5)
    
    number_of_high = 0
    for x in correlations:
        if x > 0.5:
            number_of_high += 1
    
    return number_of_high


# Returns the sparcity level of the dataset
@feature_extractor
def sparcity_level(X, y=None):
    valuated = 0
    for row in X:
        for data in row:
            if data is not None:
                valuated += 1
    return valuated / (len(X) * len(X[0]))

    
####################################### MULTIVALUED META-FEATURES ###########################################

# Returns the minimum values of each numeric attribute
@feature_extractor
def minimum_values(X, y=None):
    min_values = []
    for j in range(0, len(X[0])):
        if isinstance(X[0][j], (int, float)):
            min_value = X[0][j]
            for i in range(1, len(X)):
                min_value = min(min_value, X[i][j])
            min_values.append(min_value)
    return min_values


# Returns the maximum values of each numeric attribute
@feature_extractor
def maximum_values(X, y=None):
    max_values = []
    for j in range(0, len(X[0])):
        if isinstance(X[0][j], (int, float)):
            max_value = X[0][j]
            for i in range(1, len(X)):
                max_value = max(max_value, X[i][j])
            max_values.append(max_value)
    return max_values


# Returns the range lenght of each numeric attribute
@feature_extractor
def range_lenghts(X, y=None):
    range_values = []
    for j in range(0, len(X[0])):
        if isinstance(X[0][j], (int, float)):
            min_value = X[0][j]
            max_value = X[0][j]
            for i in range(1, len(X)):
                min_value = min(min_value, X[i][j])
                max_value = max(max_value, X[i][j])
            range_values.append(max_value - min_value)
    return range_values


# Returns the mean values of each numeric attribute
@feature_extractor
def mean_values(X, y=None):
    mean_values = []
    for j in range(0, len(X[0])):
        if isinstance(X[0][j], (int, float)):
            val_sum = 0
            for i in range(0, len(X)):
                val_sum += X[i][j]
            mean_values.append(val_sum / len(X))
    return mean_values


# Returns the trimmed mean values of each numeric attribute, which is the arithmetic 
# mean excluding the 20% of the lowest and highest instances
@feature_extractor
def trimmed_values(X, y=None):
    trimmed_mean_values = []
    for j in range(0, len(X[0])):
        if isinstance(X[0][j], (int, float)):
            values = []
            for i in range(0, len(X)):
                values.append(X[i][j])
            values.sort()
            20_percent = len(values) / 5
            val_sum = 0
            for x in values[20_percent : len(values) - 20_percent]:
                val_sum += x
            trimmed_mean_values.append(val_sum / (len(values) - 2 * 20_percent))
    return trimmed_mean_values


# Returns the median values of each numeric attribute
@feature_extractor
def median_values(X, y=None):
    median_values = []
    for j in range(0, len(X[0])):
        if isinstance(X[0][j], (int, float)):
            values = []
            for i in range(0, len(X)):
                values.append(X[i][j])
            values.sort()
            if len(values) % 2 != 0:
                median.append(values[len(values) / 2])
            else:
                median.append((values[len(values) / 2 + 1] + values[len(values) / 2]) / 2)
    return median_values


# Returns the variance values of each numeric attribute
@feature_extractor
def variance_values(X, y=None):
    variances = []
    for j in range(0, len(X[0])):
        if isinstance(X[0][j], (int, float)):
            squares_sum = 0
            mean = 0
            for i in range(0, len(X)):
                squares_sum += X[i][j] ** 2
                mean += X[i][j]
            mean /= len(X)
            variances.append(squares_sum / len(X) - mean ** 2)
    return variances


# Returns the standard deviation values of each numeric attribute
@feature_extractor
def standard_deviation_values(X, y=None):
    standard_desviations = []
    for j in range(0, len(X[0])):
        if isinstance(X[0][j], (int, float)):
            squares_sum = 0
            mean = 0
            for i in range(0, len(X)):
                squares_sum += X[i][j] ** 2
                mean += X[i][j]
            mean /= len(X)
            variance = squares_sum / len(X) - mean ** 2
            standard_desviations.append(variance ** 0.5)
    return standard_desviations


# Returns the skewness of numeric attributes.
@feature_extractor
def attributes_skewness(X, y=None):
    numeric_attributes = []
    for i in range(0, len(X[0])):
        if isinstance(X[0][i], (int, float))
            numeric_attributes.append(i)
    
    means = []
    variances = []
    standard_desviations = []
    for i in range(0, len(numeric_attributes)):
        mean = 0
        squares_sum = 0
        for j in range(0, len(X)):
            squares_sum += X[j][numeric_attributes[i]] ** 2
            mean += X[j][numeric_attributes[i]]
        mean /= len(X)
        means.append(mean)
        variances.append(squares_sum / len(X) - mean ** 2)
        standard_desviations.append(variances[len(variances) - 1] ** 0.5)


    skewness = []
    n = len(X)

    for i in range(0, len(numeric_attributes)):
        S_above = 0
        S_below = 0
        for j in range(0, n):
            mirror_val = (X[j][numeric_attributes[i]] - means[i]) ** 3
            if mirror_val > 0:
                S_above += mirror_val
            else:
                S_above += -mirror_val
        skewness.append((n / (standard_desviations[i] ** 3 * (n - 1) * (n - 2))) * (S_above - S_below))
    
    return skewness


# Resturns the attributes sparcity
@feature_extractor
def attributes_sparcity(X, y=None):
    attributes_sparcity = []
    for j in range(0, len(X[0]));
        valuated = 0
        for i in range(0, len(X)):
            if X[i][j] is not None:
                valuated += 1
        attributes_sparcity.append(valuated / len(X))
    return attributes_sparcity