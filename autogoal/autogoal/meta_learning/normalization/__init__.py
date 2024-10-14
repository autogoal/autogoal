import numpy as np
from abc import ABC, abstractmethod


class Normalizer(ABC):
    @abstractmethod
    def fit(self, feature_vectors: np.ndarray):
        """Fits the normalizer to the data."""
        pass

    @abstractmethod
    def transform(self, feature_vectors: np.ndarray) -> np.ndarray:
        """Transforms the data using the fitted normalizer."""
        pass

    def fit_transform(self, feature_vectors: np.ndarray) -> np.ndarray:
        """Fits to the data, then transforms it."""
        self.fit(feature_vectors)
        return self.transform(feature_vectors)


class LogNormalizer(Normalizer):
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon

    def fit(self, feature_vectors: np.ndarray):
        # No fitting necessary for log transformation
        return feature_vectors

    def transform(self, feature_vectors: np.ndarray) -> np.ndarray:
        return np.log1p(feature_vectors + self.epsilon)


class StandardScalerNormalizer(Normalizer):
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
        self.mean_ = None
        self.std_ = None

    def fit(self, feature_vectors: np.ndarray):
        self.mean_ = np.mean(feature_vectors, axis=0)
        self.std_ = np.std(feature_vectors, axis=0)

    def transform(self, feature_vectors: np.ndarray) -> np.ndarray:
        data_scaled = (feature_vectors - self.mean_) / (self.std_ + self.epsilon)
        return data_scaled


class MinMaxNormalizer(Normalizer):
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
        self.data_min_ = None
        self.data_max_ = None

    def fit(self, feature_vectors: np.ndarray):
        self.data_min_ = np.min(feature_vectors, axis=0)
        self.data_max_ = np.max(feature_vectors, axis=0)

    def transform(self, feature_vectors: np.ndarray) -> np.ndarray:
        data_scaled = (feature_vectors - self.data_min_ + self.epsilon) / (
            self.data_max_ - self.data_min_ + 2 * self.epsilon
        )
        return data_scaled
