import numpy as np
from autogoal.meta_learning.base import BaseNormalizer

class MinMaxLogNormalizer(BaseNormalizer):
    def __init__(self, log_transform: bool = False, epsilon: float = 1e-8):
        self.log_transform = log_transform
        self.epsilon = epsilon
        self.data_min_ = None
        self.data_max_ = None
    
    def fit(self, feature_vectors: np.ndarray):
        data = feature_vectors.copy()
        if self.log_transform:
            data = np.log1p(data)
        
        self.data_min_ = np.min(data, axis=0)
        self.data_max_ = np.max(data, axis=0)
    
    def transform(self, feature_vectors: np.ndarray) -> np.ndarray:
        data = feature_vectors.copy()
        if self.log_transform:
            data = np.log1p(data)
        
        # Custom Min-Max Scaling with Epsilon Adjustment
        data_scaled = (data - self.data_min_ + self.epsilon) / (self.data_max_ - self.data_min_ + 2 * self.epsilon)
        
        return data_scaled