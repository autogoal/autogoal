import numpy as np
from scipy.spatial.distance import (
    pdist,
    squareform,
    euclidean,
    cosine,
    cityblock,
    minkowski,
    chebyshev,
    mahalanobis,
)
from abc import ABC, abstractmethod


class DistanceMetric(ABC):
    @abstractmethod
    def compute(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Computes the distance between two feature vectors.

        Parameters:
        - vector1: The first feature vector.
        - vector2: The second feature vector.

        Returns:
        - The distance between the two vectors as a float.
        """
        pass

    @abstractmethod
    def compute_pairwise(self, feature_vectors: np.ndarray) -> np.ndarray:
        """
        Computes pairwise distances among a set of feature vectors.

        Parameters:
        - feature_vectors: A 2D numpy array where each row is a feature vector.

        Returns:
        - A 2D numpy array representing the pairwise distance matrix.
        """
        pass


class EuclideanDistance(DistanceMetric):

    def compute(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Computes the Euclidean distance between two feature vectors.
        """
        return euclidean(vector1, vector2)

    def compute_pairwise(self, feature_vectors: np.ndarray) -> np.ndarray:
        """
        Computes pairwise Euclidean distances among a set of feature vectors.
        """
        distances = pdist(feature_vectors, metric="euclidean")
        distance_matrix = squareform(distances)
        return distance_matrix


class CosineDistance(DistanceMetric):

    def compute(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Computes the Euclidean distance between two feature vectors.
        """
        return cosine(vector1, vector2)

    def compute_pairwise(self, feature_vectors: np.ndarray) -> np.ndarray:
        distances = pdist(feature_vectors, metric="cosine")
        distance_matrix = squareform(distances)
        return distance_matrix


class ManhattanDistance(DistanceMetric):

    def compute(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Computes the Euclidean distance between two feature vectors.
        """
        return cityblock(vector1, vector2)

    def compute_pairwise(self, feature_vectors: np.ndarray) -> np.ndarray:
        distances = pdist(feature_vectors, metric="cityblock")
        distance_matrix = squareform(distances)
        return distance_matrix


class MinkowskiDistance(DistanceMetric):
    def __init__(self, p: int = 3):
        self.p = p

    def compute(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Computes the Euclidean distance between two feature vectors.
        """
        return minkowski(vector1, vector2)

    def compute_pairwise(self, feature_vectors: np.ndarray) -> np.ndarray:
        distances = pdist(feature_vectors, metric="minkowski", p=self.p)
        distance_matrix = squareform(distances)
        return distance_matrix


class ChebyshevDistance(DistanceMetric):

    def compute(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Computes the Euclidean distance between two feature vectors.
        """
        return chebyshev(vector1, vector2)

    def compute_pairwise(self, feature_vectors: np.ndarray) -> np.ndarray:
        distances = pdist(feature_vectors, metric="chebyshev")
        distance_matrix = squareform(distances)
        return distance_matrix


class MahalanobisDistance(DistanceMetric):
    def compute(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Computes the Euclidean distance between two feature vectors.
        """
        return mahalanobis(vector1, vector2)

    def compute_pairwise(self, feature_vectors: np.ndarray) -> np.ndarray:
        # Compute the inverse covariance matrix
        VI = np.linalg.inv(np.cov(feature_vectors.T))
        distances = pdist(feature_vectors, metric="mahalanobis", VI=VI)
        distance_matrix = squareform(distances)
        return distance_matrix
