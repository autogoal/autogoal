# In autogoal/meta_learning/feature_extractors.py

import numpy as np
import psutil
import platform
from autogoal.meta_learning.base import BaseFeatureExtractor

class SystemFeatureExtractor(BaseFeatureExtractor):
    def extract_features(self, *args, **kwargs) -> np.ndarray:
        """
        Extracts system computational resource features relevant for running LLMs.
        
        Returns:
            feature_vector (np.ndarray): A numpy array containing the system features.
        """
        # Number of CPU cores
        num_cpu_cores = psutil.cpu_count(logical=False)  # Physical cores
        
        # Total system RAM in GB
        total_system_ram = psutil.virtual_memory().total / (1024 ** 3)
        
        # GPU information
        try:
            import torch
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                gpu_memories = []
                for i in range(num_gpus):
                    gpu_properties = torch.cuda.get_device_properties(i)
                    gpu_memory = gpu_properties.total_memory / (1024 ** 3)  # Convert to GB
                    gpu_memories.append(gpu_memory)
                total_gpu_memory = sum(gpu_memories)
                per_gpu_memory = np.mean(gpu_memories)
            else:
                num_gpus = 0
                total_gpu_memory = 0.0
                per_gpu_memory = 0.0
        except ImportError:
            # PyTorch is not installed
            num_gpus = 0
            total_gpu_memory = 0.0
            per_gpu_memory = 0.0
        
        # Create the feature vector
        feature_vector = np.array([
            num_cpu_cores,
            total_system_ram,
            num_gpus,
            total_gpu_memory,
            per_gpu_memory,
        ])
        
        return feature_vector
