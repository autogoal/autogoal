from pathlib import Path
import numpy as np
import os
import cloudpickle
import json
from typing import Any, Iterable, List
from autogoal.kb._semantics import SemanticType
from tempfile import mkdtemp
import shutil
from scipy.sparse import issparse, vstack

TEMP_DATA_PATH = Path.home() / ".autogoal" / "data" / "temp"

def clean_temporary_datasets():
    try:
        shutil.rmtree(TEMP_DATA_PATH)
    except FileNotFoundError:
        pass  # Directory already deleted, no action required
    except Exception as e:
        print(f"Error deleting temporary directory: {e}")

class Dataset:
    def __init__(self, semantic_type_instance: SemanticType, data: List[Any] or np.ndarray, storage_batch_size: int = 100, load_batch_size: int = 200): # type: ignore
        self.semantic_type_instance = semantic_type_instance
        self.storage_batch_size = storage_batch_size
        self.load_batch_size = load_batch_size
        
        os.makedirs(TEMP_DATA_PATH, exist_ok=True)
        self.storage_path = mkdtemp(prefix=semantic_type_instance._name(), dir=TEMP_DATA_PATH)  # Creates a temporary directory for storage
        self.metadata = {
            "type": semantic_type_instance._name(),
            "shape": None,
            "storage": [],
            "batch_size": self.storage_batch_size
        }
        self._store_data(data)
        
        # Iterator state
        self._current_load_index = 0  
        self._accumulated_batch = []
        self._current_yielded = 0

    def _store_data(self, data):
        batch_start = 0
        for i, batch in enumerate(self._batch_data(data, self.storage_batch_size)):
            batch_filename = os.path.join(self.storage_path, f"data_batch_{batch_start}-{batch_start + len(batch) - 1}.pkl")
            self.metadata['storage'].append(batch_filename)
            with open(batch_filename, 'wb') as file:
                cloudpickle.dump(batch, file)
            batch_start += len(batch)

        with open(os.path.join(self.storage_path, "metadata.json"), 'w') as meta_file:
            json.dump(self.metadata, meta_file)
            
    def _batch_data(self, data):
        for i in range(0, len(data), self.storage_batch_size):
            yield data[i:i+self.storage_batch_size]

    def load_batch(self, batch_index: int) -> List[Any] or np.ndarray: # type: ignore
        if batch_index >= len(self.metadata['storage']):
            raise IndexError("Batch index out of range.")
        batch_filename = self.metadata['storage'][batch_index]
        with open(batch_filename, 'rb') as file:
            batch_data = cloudpickle.load(file)
        return batch_data

    def load_all_data(self) -> List[Any] or np.ndarray: # type: ignore
        all_data = []
        for batch_file in self.metadata['storage']:
            with open(batch_file, 'rb') as file:
                all_data.extend(cloudpickle.load(file))
        return np.array(all_data) if self.metadata['shape'] else all_data

    def generate_batches(self, new_batch_size: int) -> Iterable[List[Any] or np.ndarray]: # type: ignore
        """Generate batches of data with the specified batch size."""
        for batch_file in self.metadata['storage']:
            with open(batch_file, 'rb') as file:
                data_batch = cloudpickle.load(file)
                for i in range(0, len(data_batch), new_batch_size):
                    yield data_batch[i:i+new_batch_size]
    
    def __iter__(self):
        self._current_load_index = 0
        self._accumulated_batch = []
        self._current_yielded = 0
        return self

    def __next__(self):
        while len(self._accumulated_batch) < self.load_batch_size and self._current_load_index < len(self.metadata['storage']):
            batch_filename = self.metadata['storage'][self._current_load_index]
            with open(batch_filename, 'rb') as file:
                mini_batch = cloudpickle.load(file)
            if issparse(mini_batch):
                self._accumulated_batch = vstack([self._accumulated_batch, mini_batch]) if self._accumulated_batch != [] else mini_batch
            else:
                self._accumulated_batch.extend(mini_batch)
            self._current_load_index += 1

        if issparse(self._accumulated_batch):
            yield_size = min(self._accumulated_batch.shape[0], self.load_batch_size)
            current_batch = self._accumulated_batch[:yield_size]
            self._accumulated_batch = self._accumulated_batch[yield_size:]
        else:
            yield_size = min(len(self._accumulated_batch), self.load_batch_size)
            current_batch = self._accumulated_batch[:yield_size]
            self._accumulated_batch = self._accumulated_batch[yield_size:]

        if current_batch.shape[0] == 0:
            raise StopIteration

        return current_batch
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            # Adjust for negative indices and bounds check
            if idx < 0:
                idx += len(self)
            if idx < 0 or idx >= len(self):
                raise IndexError("The index is out of bounds.")

            # Find the batch containing the item
            for batch_file in self.metadata['storage']:
                with open(batch_file, 'rb') as file:
                    batch = cloudpickle.load(file)
                    if issparse(batch):
                        if idx < batch.shape[0]:
                            return batch[idx].toarray()  # Convert to dense array for consistency
                        idx -= batch.shape[0]
    
    def __del__(self):
        try:
            shutil.rmtree(self.storage_path)
        except FileNotFoundError:
            pass  # Directory already deleted, no action required
        except Exception as e:
            print(f"Error deleting temporary directory: {e}")

    def __len__(self):
        return self.metadata['shape'][0]
    
class SimpleDataset():
    def __init__(self, semantic_type_instance: SemanticType, data: List[Any] or np.ndarray): # type: ignore
        self.semantic_type_instance = semantic_type_instance
        
        os.makedirs(TEMP_DATA_PATH, exist_ok=True)
        self.storage_path = mkdtemp(prefix=semantic_type_instance._name(), dir=TEMP_DATA_PATH)  # Creates a temporary directory for storage
        
        # Determine the length or size of the dataset depending on the data type
        if data is None:
            self.dataset_size = 0
        elif isinstance(data, list):
            self.dataset_size = len(data)
        else:  # This covers np.ndarray and sparse matrices
            self.dataset_size = np.shape(data)[0]
        
        self.metadata = {
            "type": semantic_type_instance._name(),
            "amount": self.dataset_size,
        }
        self._store_data(data)

    def _store_data(self, data):
        if data is None:
            return
        cloudpickle.dump(data, open(os.path.join(self.storage_path, "data.pkl"), 'wb'))
        
    def load_all_data(self):
        if self.dataset_size == 0:
            return None
        return cloudpickle.load(open(os.path.join(self.storage_path, "data.pkl"), 'rb'))
    
    def __iter__(self):
        # Load the data and create an iterator
        self._data_iter = iter(self.load_all_data())
        return self

    def __next__(self):
        # Use the iterator to yield the next item
        if self._data_iter is None:
            raise StopIteration  # In case __iter__ wasn't called
        return next(self._data_iter)
    
    def __len__(self):
        return self.metadata['amount'][0]
    
    def __del__(self):
        try:
            shutil.rmtree(self.storage_path)
        except FileNotFoundError:
            pass  # Directory already deleted, no action required
        except Exception as e:
            print(f"Error deleting temporary directory: {e}")


if __name__ == "__main__":
    import numpy as np

    # Define a mock SemanticType for testing purposes
    class SemanticType:
        @classmethod
        def _name(cls):
            return "MockSemanticType"

    # Generate random data
    np.random.seed(42)  # Ensure reproducibility
    data = range(1150)

    # Initialize the Dataset with the generated data
    dataset = Dataset(semantic_type_instance=SemanticType(), data=data, storage_batch_size=100, load_batch_size=250)

    # Serialize the Dataset
    with open('dataset.pkl', 'wb') as f:
        cloudpickle.dump(dataset, f)

    # Deserialize the Dataset
    with open('dataset.pkl', 'rb') as f:
        loaded_dataset = cloudpickle.load(f)

    # Use the loaded Dataset
    for i, batch in enumerate(loaded_dataset):
        print(f"Batch {i+1} shape: {batch.shape}")
        print(batch)
