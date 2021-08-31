# Gelin Eguinosa Rosique

from autogoal.ml import AutoML
from autogoal.contrib import find_classes
from autogoal.kb import Text, Seq
from autogoal.search import RichLogger

from translator import Translator
from _text_similarity import text_similarity
from _test_dataset import load


# Loading Data
X_train, y_train, X_test, y_test = load()

# Creating an instance of AutoML with the translator class
automl = AutoML(
    input=Seq[Text],
    output=Seq[Text],
    registry=[Translator] + find_classes(),
    score_metric=text_similarity
)

# Testing the Automl created
automl.fit(X_train, y_train, logger=RichLogger())

score = automl.score(X_test, y_test)
print(score)
