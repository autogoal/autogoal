import numpy as np
import platform
import psutil
import subprocess
import shutil
from autogoal.meta_learning.feature_extraction._base import FeatureExtractor

class SystemFeatureExtractor(FeatureExtractor):
    def __init__(self, gpu_device_id: str = None):
        self.gpu_device_id = gpu_device_id

    def extract_features(self, *args, **kwargs) -> np.ndarray:
        # CPU Information
        cpu_physical_cores = self._get_cpu_physical_cores()
        cpu_logical_cores = self._get_cpu_logical_cores()
        cpu_max_frequency = self._get_cpu_max_frequency()
        memory_total, _ = self._get_memory_info()

        # GPU Information
        gpu_total_memory = 0  # Default value if GPU info is not available

        if self.gpu_device_id is not None:
            gpu_total_memory = self._get_gpu_total_memory()

        # Create the feature vector
        feature_vector = np.array([
            cpu_physical_cores,
            cpu_logical_cores,
            cpu_max_frequency,
            memory_total,
            gpu_total_memory,
        ], dtype=np.float32)

        return feature_vector

    def _get_cpu_physical_cores(self):
        try:
            return psutil.cpu_count(logical=False) or 0
        except Exception:
            return 0

    def _get_cpu_logical_cores(self):
        try:
            return psutil.cpu_count(logical=True) or 0
        except Exception:
            return 0

    def _get_cpu_max_frequency(self):
        try:
            freq = psutil.cpu_freq()
            return freq.max / 1000.0 if freq else 0  # Convert MHz to GHz
        except Exception:
            return 0

    def _get_memory_info(self):
        try:
            virtual_mem = psutil.virtual_memory()
            total = virtual_mem.total / (1024 ** 3)  # Convert bytes to GB
            available = virtual_mem.available / (1024 ** 3)
            return total, available
        except Exception:
            return 0, 0

    def _get_gpu_total_memory(self):
        gpu_total_memory = 0
        try:
            if shutil.which('nvidia-smi') is not None:
                command = f"nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i {self.gpu_device_id}"
                result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT).decode().strip()
                gpu_total_memory = float(result) / 1024.0  # Convert MB to GB
            else:
                # nvidia-smi is not available
                gpu_total_memory = 0
        except Exception as e:
            # Handle exceptions gracefully
            gpu_total_memory = 0
        return gpu_total_memory
