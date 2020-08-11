try:
    from tensorflow import keras

    # assert keras.__version__ == "2.3.1"
except:
    print("(!) Code in `autogoal.contrib.keras` requires `keras==2.3.1`.")
    print("(!) You can install it with `pip install autogoal[keras]`.")
    raise


from ._base import (
    KerasClassifier,
    KerasSequenceClassifier,
    KerasSequenceTagger,
    KerasImageClassifier,
    KerasImagePreprocessor,
)
from ._grammars import build_grammar
