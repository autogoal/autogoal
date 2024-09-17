import numpy as np
import pandas as pd
from autogoal.meta_learning import (
    TextClassificationFeatureExtractor,
    SystemFeatureExtractor,
    MinMaxLogNormalizer,
    EuclideanDistance
)

def main():
    # Initialize feature extractors
    dataset_feature_extractor = TextClassificationFeatureExtractor()
    system_feature_extractor = SystemFeatureExtractor()
    
    # Extract system features
    print('Extracting system features...')
    system_feature_vector = system_feature_extractor.extract_features()
    
    # For demonstration, let's create a hypothetical list of system feature vectors
    # representing different systems (in practice, these would be collected from actual systems)
    system_feature_vectors = [system_feature_vector]
    system_names = ['CurrentSystem']
    
    # Example: Add another hypothetical system (e.g., a cloud instance)
    hypothetical_system = np.array([16, 64.0, 4, 48.0, 12.0])  # Example values
    system_feature_vectors.append(hypothetical_system)
    system_names.append('HypotheticalSystem')
    
    # Convert to NumPy array
    system_feature_vectors = np.array(system_feature_vectors)
    
    # Normalize the system feature vectors
    normalizer = MinMaxLogNormalizer(log_transform=True)
    normalizer.fit(system_feature_vectors)
    normalized_system_vectors = normalizer.transform(system_feature_vectors)
    
    # Compute distances between systems
    distance_metric = EuclideanDistance()
    system_distance_matrix = distance_metric.compute_distances(normalized_system_vectors)
    
    # Display the results
    print('\n===== Normalized System Feature Vectors =====')
    for name, vec in zip(system_names, normalized_system_vectors):
        print(f'{name}: {vec}')
    
    print('\n===== Pairwise System Distances (Euclidean) =====')
    distance_df = pd.DataFrame(system_distance_matrix, index=system_names, columns=system_names)
    print(distance_df)

if __name__ == '__main__':
    main()
