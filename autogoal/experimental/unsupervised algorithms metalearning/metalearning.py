import functools
import math
import numpy as np

from numpy import linalg as LA
import statistics as st
from scipy.stats import shapiro


class DatasetFeatureExtractor:
    def __init__(self, features_extractors=None):
        self.feature_extractors = list(features_extractors or _EXTRACTORS)

    def extract_features(self, X, y=None):
        features = {}

        for extractor in self.feature_extractors:
            #features.update(**extractor(X, y))
            print(extractor.__name__ + " " + extractor(X, y))

        return features
        

_EXTRACTORS = []

def feature_extractor(func):
    @functools.wraps(func)
    def wrapper(X, y=None):
        #try:
        result = func(X, y)
        #except:
        #    result = None

        return str(result)
        #return {func.__name__: result}

    _EXTRACTORS.append(wrapper)
    return wrapper
    
    
def specific_types_attributes_index(X, types):
    N, M = get_dimentions(X)
    attributes_index = []
    for j in range(0, M):
        if isinstance(X[0][j], types):
            attributes_index.append(j)
    return attributes_index


def get_dimentions(X):
    return len(X), len(X[0])


def values_from_rows(X, attr):
    values = []
    N, _ = get_dimentions(X)
    for j in attr:
        vals = []
        for i in range(0, N):
            vals.append(X[i][j])
        values.append(vals)
    return values


def calculate_means(values):
    means = []
    for i in range(0, len(values)):
        means.append(st.mean(values[i]))
    return means

def calculate_variances(values):
    variances = []
    for i in range(0, len(values)):
        variances.append(st.variance(values[i]))
    return variances


def calculate_stds(values):
    stds = []
    for i in range(0, len(values)):
        stds.append(st.stdev(values[i]))
    return stds


def calculate_medians(values):
    medians = []
    for i in range(0, len(values)):
        medians.append(st.median(values[i]))
    return medians

# Feature extractor methods

###################################### SINGLE VALUED META-FEATURES ########################################

# Returns the amount of data attributes.
@feature_extractor
def attributes_amount(X, y=None):
    N, M = get_dimentions(X)
    return M


# Returns log_2 of the amount of data attributes.
@feature_extractor
def attributes_amount_log_2(X, y=None):
    N, M = get_dimentions(X)
    return math.log2(M)


# Returns log_10 of the amount of data attributes.
@feature_extractor
def attributes_amount_log_10(X, y=None):
    N, M = get_dimentions(X)
    return math.log10(M)


# Returns the amount of continuous attributes.
@feature_extractor
def continuous_amount(X, y=None):
    N, M = get_dimentions(X)
    amount = 0
    for val in X[0]:
        if isinstance(val, (float)):
            amount += 1
    return amount

# Returns the proportion of continuous attributes.
@feature_extractor
def continuous_proportion(X, y=None):
    N, M = get_dimentions(X)
    amount = 0
    for val in X[0]:
        if isinstance(val, (float)):
            amount += 1
    return float(amount) / float(M)


# Returns the mean absolute correlation (Pearson coefficient) between continuous attributes.
@feature_extractor
def continuous_mean_absolute_correlation(X, y=None):
    N, M = get_dimentions(X)
    continuous_attributes = specific_types_attributes_index(X, (float))
    values = values_from_rows(X, continuous_attributes)

    means = calculate_means(values)
    variances = calculate_variances(values)

    correlations = []

    for i in range(0, len(values) - 1):
        for j in range(i + 1, len(values)):
            product_sum = 0
            for k in range(0, N):
                product_sum += values[i][k] * values[j][k]
            covariance = product_sum / N - means[i] * means[j]
            correlations.append(covariance / (variances[i] * variances[j]) ** 0.5)
    
    return st.mean(correlations)


# Returns the mean skewness of continuous attributes.
@feature_extractor
def continuous_mean_skewness(X, y=None):
    N, M = get_dimentions(X)
    continuous_attributes = specific_types_attributes_index(X, (int, float))
    values = values_from_rows(X, continuous_attributes)
    means = calculate_means(values)
    variances = calculate_variances(values)
    standard_desviations = calculate_stds(values)

    skewness = []

    for i in range(0, len(values)):
        S_above = 0
        S_below = 0
        for j in range(0, N):
            mirror_val = (values[i][j] - means[i]) ** 3
            if mirror_val > 0:
                S_above += mirror_val
            else:
                S_above += -mirror_val
        skewness.append((N / (standard_desviations[i] ** 3 * (N - 1) * (N - 2))) * (S_above - S_below))

    return st.mean(skewness)


# Returns the mean kurtosis of continuous attributes.
@feature_extractor
def continuous_mean_kurtosis(X, y=None):
    N, M = get_dimentions(X)
    continuous_attributes = specific_types_attributes_index(X, (int, float))
    values = values_from_rows(X, continuous_attributes)
    means = calculate_means(values)
    variances = calculate_variances(values)
    standard_desviations = calculate_stds(values)
    
    kurtosis = []
    left_coeficient = (N * (N + 1)) / ((N - 1) * (N - 2) * (N - 3))
    right_substractor = (3 * (N - 1) ** 2) / ((N - 2) * (N - 3))

    for i in range(0, len(values)):
        central_value = 0
        for j in range(0, N):
            central_value += ((values[i][j] - means[i]) ** 4) / (standard_desviations[i] ** 4)
        kurtosis.append(left_coeficient * central_value - right_substractor)
    
    return st.mean(kurtosis)


# Returns the number of attribute pairs with high correlation (Pearson coefficient)
@feature_extractor
def number_of_high_correlated_pairs(X, y=None):
    N, M = get_dimentions(X)
    numeric_attributes = specific_types_attributes_index(X, (int, float))
    values = values_from_rows(X, numeric_attributes)
    means = calculate_means(values)
    variances = calculate_variances(values)
    
    correlations = []
    for i in range(0, len(values) - 1):
        for j in range(i + 1, len(values)):
            product_sum = 0
            for k in range(0, N):
                product_sum += values[i][k] * values[j][k]
            covariance = product_sum / N - means[i] * means[j]
            correlations.append(covariance / (variances[i] * variances[j]) ** 0.5)
    
    number_of_high = 0
    for x in correlations:
        if x > 0.5:
            number_of_high += 1
    
    return number_of_high


# Returns the sparcity level of the dataset
@feature_extractor
def sparcity_level(X, y=None):
    N, M = get_dimentions(X)
    valuated = 0
    for row in X:
        for data in row:
            if data is not None:
                valuated += 1
    return valuated / (N * M)


# Returns the eigenvalues of the attributes correlation matrix
@feature_extractor
def correlation_matrix_eigenvalues(X, y = None):
    N, M = get_dimentions(X)
    numeric_attributes = specific_types_attributes_index(X, (int, float))
    values = values_from_rows(X, numeric_attributes)
    means = calculate_means(values)

    correlation_matrix = []
    for i in range(0, len(values)):
        row = []
        for j in range(0, len(values)):
            product_sum = 0
            for k in range(0, N):
                product_sum += X[k][numeric_attributes[i]] * X[k][numeric_attributes[j]]
            row.append(product_sum / N - means[i] * means[j])
        correlation_matrix.append(row)

    w, v = LA.eig(np.array(correlation_matrix))
    
    
    eigenvalues = []
    for e in w:
        eigenvalues.append(e)

    return eigenvalues


# Returns the number of atributes with ouliers values
@feature_extractor
def attributes_with_outliers(X, y = None):
    N, M = get_dimentions(X)
    numeric_attributes = specific_types_attributes_index(X, (int, float))
    values = values_from_rows(X, numeric_attributes)

    result = 0

    for i in range(0, len(values)):
        has_outliers = False
        threshold = 3
        mean = st.mean(values[i])
        standard_deviation = st.stdev(values[i])

        for x in values[i]:
            z_score = (x - mean) / standard_deviation
            if np.abs(z_score) > threshold:
                has_outliers = True
                break
        
        if has_outliers:
            result += 1
    
    return result


# Returns the number of attributes with a normal distribution (with Shapiro-Wilk test)
@feature_extractor
def normal_distributed_count(X, y=None):
    N, M = get_dimentions(X)
    numeric_attributes = specific_types_attributes_index(X, (int, float))
    values = values_from_rows(X, numeric_attributes)

    result = 0

    for i in range(0, len(values)):
        stat, p = shapiro(np.array(values[i]))
        alpha = 0.05
        if p > alpha:
            result += 1
    
    return result

####################################### MULTIVALUED META-FEATURES ###########################################

# Returns the minimum values of each numeric attribute
@feature_extractor
def minimum_values(X, y=None):
    N, M = get_dimentions(X)
    min_values = []
    for j in range(0, M):
        if isinstance(X[0][j], (int, float)):
            min_value = X[0][j]
            for i in range(1, N):
                min_value = min(min_value, X[i][j])
            min_values.append(min_value)
    return min_values


# Returns the maximum values of each numeric attribute
@feature_extractor
def maximum_values(X, y=None):
    N, M = get_dimentions(X)
    max_values = []
    for j in range(0, M):
        if isinstance(X[0][j], (int, float)):
            max_value = X[0][j]
            for i in range(1, N):
                max_value = max(max_value, X[i][j])
            max_values.append(max_value)
    return max_values


# Returns the range lenght of each numeric attribute
@feature_extractor
def range_lenghts(X, y=None):
    N, M = get_dimentions(X)
    range_values = []
    for j in range(0, M):
        if isinstance(X[0][j], (int, float)):
            min_value = X[0][j]
            max_value = X[0][j]
            for i in range(1, N):
                min_value = min(min_value, X[i][j])
                max_value = max(max_value, X[i][j])
            range_values.append(max_value - min_value)
    return range_values


# Returns the mean values of each numeric attribute
@feature_extractor
def mean_values(X, y=None):
    N, M = get_dimentions(X)
    numeric_attributes = specific_types_attributes_index(X, (int, float))
    values = values_from_rows(X, numeric_attributes)
    means = calculate_means(values)
    return means


# Returns the trimmed mean values of each numeric attribute, which is the arithmetic 
# mean excluding the 20% of the lowest and highest instances
@feature_extractor
def trimmed_values(X, y=None):
    N, M = get_dimentions(X)
    numeric_attributes = specific_types_attributes_index(X, (int, float))
    values = values_from_rows(X, numeric_attributes)
    trimmed_mean_values = []
    for x in values:
        x.sort()
        tw_percent = int(len(x) / 5)
        val_sum = 0
        for z in x[tw_percent : len(x) - tw_percent]:
            val_sum += z
        trimmed_mean_values.append(val_sum / (len(x) - 2 * tw_percent))
    return trimmed_mean_values


# Returns the median values of each numeric attribute
@feature_extractor
def median_values(X, y=None):
    N, M = get_dimentions(X)
    numeric_attributes = specific_types_attributes_index(X, (int, float))
    values = values_from_rows(X, numeric_attributes)
    medians = calculate_medians(values)
    return medians


# Returns the variance values of each numeric attribute
@feature_extractor
def variance_values(X, y=None):
    N, M = get_dimentions(X)
    numeric_attributes = specific_types_attributes_index(X, (int, float))
    values = values_from_rows(X, numeric_attributes)
    variances = calculate_variances(values)
    return variances


# Returns the standard deviation values of each numeric attribute
@feature_extractor
def standard_deviation_values(X, y=None):
    N, M = get_dimentions(X)
    numeric_attributes = specific_types_attributes_index(X, (int, float))
    values = values_from_rows(X, numeric_attributes)
    standard_desviations = calculate_stds(values)
    return standard_desviations


# Returns the skewness of numeric attributes.
@feature_extractor
def attributes_skewness(X, y=None):
    N, M = get_dimentions(X)
    numeric_attributes = specific_types_attributes_index(X, (int, float))
    values = values_from_rows(X, numeric_attributes)
    
    means = calculate_means(values)
    variances = calculate_variances(values)
    standard_desviations = calculate_stds(values)

    skewness = []

    for i in range(0, len(values)):
        S_above = 0
        S_below = 0
        for j in range(0, N):
            mirror_val = (values[i][j] - means[i]) ** 3
            if mirror_val > 0:
                S_above += mirror_val
            else:
                S_above += -mirror_val
        skewness.append((N / (standard_desviations[i] ** 3 * (N - 1) * (N - 2))) * (S_above - S_below))
    
    return skewness


# Resturns the attributes sparcity
@feature_extractor
def attributes_sparcity(X, y=None):
    N, M = get_dimentions(X)
    attributes_sparcity = []
    for j in range(0, M):
        valuated = 0
        for i in range(0, N):
            if X[i][j] is not None:
                valuated += 1
        attributes_sparcity.append(valuated / N)
    return attributes_sparcity


# Returns the number of outlier values of each numeric atribute
@feature_extractor
def attributes_outliers(X, y = None):
    N, M = get_dimentions(X)
    numeric_attributes = specific_types_attributes_index(X, (int, float))
    values = values_from_rows(X, numeric_attributes)
    result = []

    means = calculate_means(values)
    standard_deviations = calculate_stds(values)

    for i in range(0, len(values)):
        outlier_number = 0
        threshold = 3
        for x in values[i]:
            z_score = (x - means[i]) / standard_deviations[i]
            if np.abs(z_score) > threshold:
                outlier_number += 1
        
        result.append(outlier_number)
    
    return result