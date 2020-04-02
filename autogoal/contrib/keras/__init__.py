try:
    from tensorflow import keras

    # assert keras.__version__ == "2.3.1"
except:
    print("(!) Code in `autogoal.contrib.keras` requires `keras==2.3.1`.")
    print("(!) You can install it with `pip install autogoal[keras]`.")
    raise


from ._base import (
    KerasNeuralNetwork,
    KerasClassifier,
    KerasSequenceClassifier,
    KerasSequenceTagger,
    KerasImageClassifier,
    KerasImagePreprocessor,
)
from ._grammars import build_grammar


def find_classes(self, include=None, exclude=None):
    return [
        KerasSequenceClassifier,
        KerasClassifier,
        KerasSequenceTagger,
        KerasImageClassifier,
    ]
