from autogoal.contrib.keras._base import KerasNeuralNetwork
from autogoal.contrib.keras._grammars import generate_grammar
from autogoal.kb import Supervised
from autogoal.kb._semantics import Seq

from ..segmentation._semantics import Image, ImageMask
from ._grammar import Modules

from tensorflow.keras.layers import Input, concatenate, Conv2D


class KerasImageSegmenter(KerasNeuralNetwork):
    
    """
    Class that represents an Image Segmenter with a Keras-based CNN architecture. 
    """
    def __init__(self, **kwargs):
        super().__init__(grammar=self._build_grammar(), optimizer="adam", **kwargs)

    def _build_grammar(self):
        return generate_grammar(Modules.ConvNN())

    def _build_input(self, X):
        return Input(shape=X.shape[1:])

    def _build_output_layer(self, y):
        self._num_classes = y.shape[1]

        if "loss" not in self._compile_kwargs:
            self._compile_kwargs["loss"] = "sparse_categorical_crossentropy"
            self._compile_kwargs["metrics"] = ["accuracy"]

        return Conv2D(3, 3, activation='softmax', padding='same')

    def _build_output(self, outputs, y):
        if len(outputs) > 1:
            outputs = concatenate(outputs)
        else:
            outputs = outputs[0]

        return self._build_output_layer(y)(outputs)

    def run(self, X: Seq[Image], y: Supervised[Seq[ImageMask]]) -> Seq[ImageMask]:
        return super().run(X, y)
