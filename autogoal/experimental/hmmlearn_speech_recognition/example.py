from autogoal.ml import AutoML
from autogoal.kb import *
from autogoal.search import RichLogger
from autogoal.utils import Min, Gb
from autogoal.experimental.hmmlearn_speech_recognition.util import AudioFile
from autogoal.experimental.hmmlearn_speech_recognition._manual import HMMLearnSpeechRecognizer
from autogoal.contrib import find_classes
import numpy as np
from os import listdir
from autogoal.experimental.hmmlearn_speech_recognition.dataset import load

automl = AutoML(
    input=(Seq[AudioFile], Supervised[Seq[Word]]),
    output=Seq[Word],
    registry=[HMMLearnSpeechRecognizer] + find_classes(),
    evaluation_timeout= 2 * Min,
    memory_limit= 4 * Gb,
    search_timeout= 5 * Min
)

X_train, y_train, X_test, y_test = load()
print(y_test)
automl.fit(X_train, y_train, logger=[RichLogger()])
score = automl.score(X_test, y_test)
print(f'Score: {score}' )
answers = automl.predict(X_test)
print(f'Answers: {answers}')