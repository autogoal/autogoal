from sparseml.keras.optim import ScheduledModifierManager
from tensorflow.keras.utils import to_categorical
from autogoal.contrib.keras  import KerasClassifier
from autogoal.grammar import CategoricalValue


def build_sparseml_keras_classifier(path_to_recipe: str):
    """Build custom KerasClassifier algorithm applaying sparcification techniques provided in path_to_recipe yaml file."""
    class SparseMLKerasClassifier(KerasClassifier):
        def __init__(self,optimizer: CategoricalValue("sgd", "adam", "rmsprop"), grammar=None, **kwargs) -> None:
            self._path=path_to_recipe
            self._manager=ScheduledModifierManager.from_yaml(self._path)
            super().__init__(grammar=grammar or self._build_grammar(), optimizer=optimizer, **kwargs)

        def fit(self,X,y):
            self._classes = {k: v for k, v in zip(set(y), range(len(y)))}
            self._inverse_classes = {v: k for k, v in self._classes.items()}
            y = [self._classes[yi] for yi in y]
            y = to_categorical(y)

            #Create Model
            if self._graph is None:
                raise TypeError("You must call `sample` to generate the internal model.")

            self._build_nn(self._graph, X, y)

            #Sparcificate Model
            old_model = self._model
            self._model = self._manager.finalize(old_model)
            self._model.compile(**self._compile_kwargs)

            #Fit Model
            self._fit_model(X, y) 
            
    
    return SparseMLKerasClassifier
        