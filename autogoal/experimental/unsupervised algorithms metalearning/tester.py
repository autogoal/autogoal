import metalearning as mtl
from dataset_examples.airlines_safety import data as airlines_data
from dataset_examples.hate_crimes import data as crimes_data


de = mtl.DatasetFeatureExtractor()

airlines_meta_features = de.extract_features(airlines_data)
crimes_meta_features = de.extract_features(crimes_data)

def sample_print(name, meta_features):
    result = ""
    print(name + "\n" + "-----------------------------------\n")
    for k in meta_features.keys():
        result += k + "\n" + str(meta_features[k]) + "\n\n"
    return result

print(sample_print("airlines_safety", airlines_meta_features))
print(sample_print("hate_crimes", crimes_meta_features))