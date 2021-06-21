import metalearning as mtl
from dataset_examples.airlines_safety import data as airlines_data
from dataset_examples.hate_crimes import data as crimes_data


de = mtl.DatasetFeatureExtractor()

airlines_meta_features = de.extract_features(airlines_data)
crimes_meta_features = de.extract_features(crimes_data)

# take the returning object and prints it in a nicer way
def sample_print(name, meta_features):
    result = ""
    print(name + "\n" + "-----------------------------------\n")
    for k in meta_features.keys():
        result += k + "\n" + str(meta_features[k]) + "\n\n"
    return result

print(sample_print("airlines_safety", airlines_meta_features))

# SHOULD PRINT:

# airlines_safety
# -----------------------------------

# instances_amount
# 56

# attributes_amount
# 8

# attributes_amount_log_2
# 3.0

# attributes_amount_log_10
# 0.9030899869919435

# examples_amount
# 0

# examples_amount_log_2
# 0

# examples_amount_log_10
# 0

# binary_amount
# 0

# binary_proportion
# 0.0

# categorical_amount
# 1

# categorical_proportion
# 0.125

# discrete_amount
# 7

# discrete_proportion
# 0.875

# continuous_amount
# 7

# continuous_proportion
# 0.875

# examples_by_attributes_ratio
# 0

# missing_values_percentage
# 0.0

# continuous_mean_absolute_correlation
# 0.38967953773317343

# continuous_mean_skewness
# 2.6796087415730026

# continuous_mean_kurtosis
# 7.796108978878376

# number_of_high_correlated_pairs
# 6

# sparcity_level
# 1.0

# correlation_matrix_eigenvalues
# [2.1088110926209216e+18, 481290746256.9356, -6.182433037738022e-13, -2.642743119754036e-12, 3.4230323606506794e-13, -1.3553010373094273e-13, 1.5585162366100498e-14]

# attributes_with_outliers
# 5

# normal_distributed_count
# 0

# discrete_mean_entropy
# 128.7856419025837

# minimum_values
# [259373346, 0, 0, 0, 0, 0, 0]

# maximum_values
# [7139291291, 76, 14, 535, 24, 3, 537]

# range_lenghts
# [6879917945, 76, 14, 535, 24, 3, 537]

# mean_values
# [1384621304.732143, 7.178571428571429, 2.1785714285714284, 112.41071428571429, 4.125, 0.6607142857142857, 55.517857142857146]

# trimmed_values
# [964477153.5, 4.5588235294117645, 1.411764705882353, 66.88235294117646, 3.1176470588235294, 0.4117647058823529, 15.029411764705882]

# median_values
# [802908893.0, 4.0, 1.0, 48.5, 3.0, 0.0, 0.0]

# variance_values
# [2.1471536025282097e+18, 121.78571428571428, 8.185714285714285, 21518.282792207792, 20.65681818181818, 0.7373376623376624, 12394.981493506493]

# standard_deviation_values
# [1465316894.9166627, 11.035656495456637, 2.861068731385928, 146.69111354205404, 4.544977247667823, 0.8586836800228955, 111.33275121682071]

# attributes_skewness
# [2.6148285885802784, 4.863127516143572, 2.621775293528285, 1.748492356379912, 2.6078968554909867, 1.4569951117927993, 2.844145469095184]

# attributes_sparcity
# [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# attributes_outliers
# [2, 1, 2, 0, 1, 0, 2]

print(sample_print("hate_crimes", crimes_meta_features))

# SHOULD PRINT

# hate_crimes
# -----------------------------------

# instances_amount
# 51

# attributes_amount
# 12

# attributes_amount_log_2
# 3.584962500721156

# attributes_amount_log_10
# 1.0791812460476249

# examples_amount
# 0

# examples_amount_log_2
# 0

# examples_amount_log_10
# 0

# binary_amount
# 0

# binary_proportion
# 0.0

# categorical_amount
# 1

# categorical_proportion
# 0.08333333333333333

# discrete_amount
# 2

# discrete_proportion
# 0.16666666666666666

# continuous_amount
# 11

# continuous_proportion
# 0.9166666666666666

# examples_by_attributes_ratio
# 0

# missing_values_percentage
# 1.3071895424836601

# continuous_mean_absolute_correlation
# -0.07387078512368404

# continuous_mean_skewness
# 1.891462828957516

# continuous_mean_kurtosis
# 2.2775926163660087

# number_of_high_correlated_pairs
# 5

# sparcity_level
# 0.9869281045751634

# correlation_matrix_eigenvalues
# [(68878293.7636168+0j), (14255111.163765393+0j), (0.0678660697806088+0j), (0.004213307682170008+0j), (0.00010560141788935415+0j), (-1.0400088004021487e-16+5.89823281140603e-16j), (-1.0400088004021487e-16-5.89823281140603e-16j), (-9.879897113761415e-17+2.0020361298194214e-16j), (-9.879897113761415e-17-2.0020361298194214e-16j), (3.383245540380336e-17+0j), (-1.8080686122183686e-17+0j)]

# attributes_with_outliers
# 5

# normal_distributed_count
# 4

# discrete_mean_entropy
# 22.88721875540867

# minimum_values
# [35521, 0.028, 0.31, 0.799, 0.01, 0.04, 0.419, 0.06, 0.04, 0.067446801, 0.26694076]

# maximum_values
# [76165, 0.073, 1, 0.918, 0.13, 0.17, 0.532, 0.81, 0.7, 1.52230172, 10.95347971]

# range_lenghts
# [40644, 0.045, 0.69, 0.119, 0.12000000000000001, 0.13, 0.11300000000000004, 0.75, 0.6599999999999999, 1.454854919, 10.68653895]

# mean_values
# [55223.60784313725, 0.04956862745098039, 0.7501960784313726, 0.8691176470588236, 0.05458333333333333, 0.09176470588235294, 0.45376470588235296, 0.3156862745098039, 0.49, 0.3040929699574468, 2.36761300684]

# trimmed_values
# [54793.25806451613, 0.0495483870967742, 0.7706451612903226, 0.8723548387096772, 0.050333333333333355, 0.08967741935483874, 0.45303225806451614, 0.29903225806451605, 0.4964516129032258, 0.246089652724138, 2.1503894770999996]

# median_values
# [54916, 0.051, 0.79, 0.874, 0.045, 0.09, 0.454, 0.28, 0.49, 0.226197105, 1.987068232]

# variance_values
# [84796070.20313725, 0.00011445019607843138, 0.03297396078431372, 0.0011609858823529422, 0.0009657801418439716, 0.0006108235294117648, 0.0004364235294117649, 0.02719701960784314, 0.014091999999999999, 0.06386165350541183, 2.938635803169832]

# standard_deviation_values
# [9208.478169770358, 0.010698139841973995, 0.1815873365196861, 0.03407324290925274, 0.031077003424461177, 0.02471484431291779, 0.020890752246191732, 0.16491518913624403, 0.1187097300140136, 0.25270863361866336, 1.714244965916433]

# attributes_skewness
# [1.5425566643585955, 1.4761705973769867, 1.5503826696019152, 1.3697159488170094, 1.3799539976882687, 1.8171155591897006, 2.0282193224943628, 1.6513326879938794, 2.011030905131453, 2.8815302155157365, 3.0980825503647664]

# attributes_sparcity
# [1.0, 1.0, 1.0, 1.0, 1.0, 0.9411764705882353, 1.0, 1.0, 1.0, 1.0, 0.9215686274509803, 0.9803921568627451]

# attributes_outliers
# [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1]