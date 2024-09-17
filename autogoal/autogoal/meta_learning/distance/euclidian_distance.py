# autogoal/meta_learning/distance_metrics.py

import numpy as np
from scipy.spatial.distance import pdist, squareform
from autogoal.meta_learning.base import BaseDistanceMetric

class EuclideanDistance(BaseDistanceMetric):
    def compute_distances(self, feature_vectors: np.ndarray) -> np.ndarray:
        distances = pdist(feature_vectors, metric='euclidean')
        distance_matrix = squareform(distances)
        return distance_matrix

class CosineDistance(BaseDistanceMetric):
    def compute_distances(self, feature_vectors: np.ndarray) -> np.ndarray:
        distances = pdist(feature_vectors, metric='cosine')
        distance_matrix = squareform(distances)
        return distance_matrix

class ManhattanDistance(BaseDistanceMetric):
    def compute_distances(self, feature_vectors: np.ndarray) -> np.ndarray:
        distances = pdist(feature_vectors, metric='cityblock')
        distance_matrix = squareform(distances)
        return distance_matrix

class MinkowskiDistance(BaseDistanceMetric):
    def __init__(self, p: int = 3):
        self.p = p
    
    def compute_distances(self, feature_vectors: np.ndarray) -> np.ndarray:
        distances = pdist(feature_vectors, metric='minkowski', p=self.p)
        distance_matrix = squareform(distances)
        return distance_matrix

class ChebyshevDistance(BaseDistanceMetric):
    def compute_distances(self, feature_vectors: np.ndarray) -> np.ndarray:
        distances = pdist(feature_vectors, metric='chebyshev')
        distance_matrix = squareform(distances)
        return distance_matrix

class MahalanobisDistance(BaseDistanceMetric):
    def compute_distances(self, feature_vectors: np.ndarray) -> np.ndarray:
        # Compute the inverse covariance matrix
        VI = np.linalg.inv(np.cov(feature_vectors.T))
        distances = pdist(feature_vectors, metric='mahalanobis', VI=VI)
        distance_matrix = squareform(distances)
        return distance_matrix
