from autogoal.utils import nice_repr
from autogoal.contrib.keras._base import KerasNeuralNetwork
from autogoal.contrib.keras._grammars import generate_grammar
from ._grammar import Modules
from tensorflow.keras.layers import Input, Dense, concatenate
from autogoal.kb._semantics import MatrixContinuousDense, VectorCategorical
from autogoal.kb._algorithm import Supervised


class KerasImageSegmenter(KerasNeuralNetwork):
    def __init__(self, **kwargs):
        super().__init__(grammar=self._build_grammar(), **kwargs)

    def _build_grammar(self):
        return generate_grammar(Modules.ImageSegmenter())

    def _build_input(self, X):
        return Input(shape=X.shape[1:])

    def _build_output_layer(self, y):
        self._num_classes = y.shape[1]

        if "loss" not in self._compile_kwargs:
            self._compile_kwargs["loss"] = "categorical_crossentropy"
            self._compile_kwargs["metrics"] = ["accuracy"]

        return Dense(units=self._num_classes, activation="softmax")

    def _build_output(self, outputs, y):
        if len(outputs) > 1:
            outputs = concatenate(outputs)
        else:
            outputs = outputs[0]

        return self._build_output_layer(y)(outputs)
    
    def run(self, X: MatrixContinuousDense, y: Supervised[MatrixContinuousDense]
            ) -> MatrixContinuousDense:
        return super().run(X, y)
