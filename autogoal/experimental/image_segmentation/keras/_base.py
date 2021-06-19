from autogoal.contrib.keras._base import KerasNeuralNetwork
from autogoal.contrib.keras._grammars import generate_grammar
from ._grammar import Modules
from tensorflow.keras.layers import Input, Dense, concatenate

from autogoal.kb import Supervised, Seq
from ..segmentation._semantics import Image, ImageMask


class KerasImageSegmenter(KerasNeuralNetwork):
    def __init__(self, **kwargs):
        super().__init__(grammar=self._build_grammar(), **kwargs)

    def _build_grammar(self):
        return generate_grammar(Modules.ConvNN())

    def _build_input(self, X):
        return Input(shape=X.shape[1:])

    def _build_output_layer(self, y):
        self._num_classes = y.shape[1]

        if "loss" not in self._compile_kwargs:
            self._compile_kwargs["loss"] = "categorical_crossentropy"
            self._compile_kwargs["metrics"] = ["accuracy"]

        return Dense(units=2, activation="softmax")

    def _build_output(self, outputs, y):
        if len(outputs) > 1:
            outputs = concatenate(outputs)
        else:
            outputs = outputs[0]

        return self._build_output_layer(y)(outputs)

    def run(self, X: Seq[Image], y: Supervised[Seq[ImageMask]]) -> Seq[ImageMask]:
        return super().run(X, y)
