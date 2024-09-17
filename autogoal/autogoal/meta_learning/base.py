from abc import ABC, abstractmethod
import numpy as np

class BaseFeatureExtractor(ABC):
    @abstractmethod
    def extract_features(self, X_train: any, y_train: any, X_test: any, y_test: any) -> np.ndarray:
        """Extracts feature vector from the dataset."""
        pass

class BaseNormalizer(ABC):
    @abstractmethod
    def fit(self, feature_vectors: np.ndarray):
        """Fits the normalizer to the data."""
        pass

    @abstractmethod
    def transform(self, feature_vectors: np.ndarray) -> np.ndarray:
        """Transforms the data using the fitted normalizer."""
        pass

class BaseDistanceMetric(ABC):
    @abstractmethod
    def compute_distances(self, feature_vectors: np.ndarray) -> np.ndarray:
        """Computes pairwise distances between feature vectors."""
        pass
