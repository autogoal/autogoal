from autogoal.experimental.sparseml.util import KerasModel
from autogoal.kb import AlgorithmBase
from sparseml.keras.optim import ScheduledModifierManager

def get_sparseml_keras_algorithm(path: str):
    class SparsemlKerasAlgorithm(AlgorithmBase):
        def __init__(self):
            self._path = path
            self._manager = ScheduledModifierManager.from_yaml(self._path)
        
        def run(self, model: KerasModel) -> KerasModel:
            return self._manager.finalize(model)
