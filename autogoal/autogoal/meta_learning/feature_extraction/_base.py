import numpy as np
from abc import ABC, abstractmethod

class FeatureExtractor(ABC):
    @abstractmethod
    def extract_features(self, X_train: any = None, y_train: any = None, X_test: any = None, y_test: any = None) -> np.ndarray:
        """Extracts feature vector from the dataset."""
        pass

