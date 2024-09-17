import numpy as np
import pandas as pd
from autogoal.datasets import ag_news, imdb_50k_movie_reviews, yelp_reviews, rotten_tomatoes
from autogoal.meta_learning import (
    TextClassificationFeatureExtractor,
    MinMaxLogNormalizer,
    EuclideanDistance
)

def load_dataset(dataset):
    X_train, y_train, X_test, y_test = dataset.load(True)
    return X_train, y_train, X_test, y_test

def main():
    datasets = {
        'AGNews': ag_news,
        'IMDB': imdb_50k_movie_reviews,
        'YelpReviews': yelp_reviews,
    }
    
    feature_extractor = TextClassificationFeatureExtractor()
    feature_vectors = []
    dataset_names = []
    
    # Extract features for each dataset
    for name, dataset in datasets.items():
        print(f'Processing dataset: {name}')
        X_train, y_train, X_test, y_test = load_dataset(dataset)
        feature_vector = feature_extractor.extract_features(X_train, y_train, X_test, y_test)
        feature_vectors.append(feature_vector)
        dataset_names.append(name)
    
    # Process the new dataset
    print('\nProcessing new target dataset: Rotten Tomatoes')
    # Replace with actual new dataset loading
    X_train_new, y_train_new, X_test_new, y_test_new = load_dataset(rotten_tomatoes)
    feature_vector_new = feature_extractor.extract_features(X_train_new, y_train_new, X_test_new, y_test_new)
    feature_vectors.append(feature_vector_new)
    dataset_names.append('RottenTomatoes')
    
    # Convert feature vectors to numpy array
    feature_vectors = np.array(feature_vectors)
    
    # Normalize all feature vectors together
    normalizer = MinMaxLogNormalizer(log_transform=True)
    normalizer.fit(feature_vectors)
    normalized_vectors = normalizer.transform(feature_vectors)
    
    # Compute distances
    distance_metric = EuclideanDistance()
    distance_matrix = distance_metric.compute_distances(normalized_vectors)
    
    # Display the results
    print('\n===== Normalized Feature Vectors =====')
    for name, vec in zip(dataset_names, normalized_vectors):
        print(f'{name}: {vec}')
    
    print('\n===== Pairwise Dataset Distances (Euclidean) =====')
    distance_df = pd.DataFrame(distance_matrix, index=dataset_names, columns=dataset_names)
    print(distance_df)

if __name__ == '__main__':
    main()
