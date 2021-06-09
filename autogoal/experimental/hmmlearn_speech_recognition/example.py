from autogoal.ml import AutoML
from autogoal.kb import *
from autogoal.search import RichLogger
from autogoal.utils import Min, Gb
from autogoal.experimental.hmmlearn_speech_recognition.util import AudioFile
from autogoal.experimental.hmmlearn_speech_recognition._manual import HMMLearnSpeechRecognizer
from autogoal.contrib import find_classes
import numpy as np
from os import listdir

automl = AutoML(
    input=(Seq[AudioFile], Supervised[Seq[Word]]),
    output=Seq[Word],
    registry=[HMMLearnSpeechRecognizer],
    evaluation_timeout= Min,
    memory_limit= 4 * Gb,
    search_timeout= Min
)

root_dir = './autogoal/experimental/hmmlearn_speech_recognition'

X_train, y_train, X_test, y_test = [], [], [], []
for subfolder in listdir(f'{root_dir}/data'):
    base_path = f'{root_dir}/data/{subfolder}/'
    for audio_file in listdir(base_path)[:-1]:
        file_path = base_path + audio_file
        X_train.append(file_path)
        y_train.append(subfolder)
    test_audio_file = base_path + listdir(base_path)[-1]
    X_test.append(test_audio_file)
    y_test.append(subfolder)

print('Train arguments:')
print(f'X({len(X_train)}): \n{enumerate(X_train)}')
print(f'y({len(y_train)}):\n{y_train}\n\n')

# speech_recognizer = HMMLearnSpeechRecognizer(4, 1000)
# speech_recognizer._train(X_train, y_train)
# print(speech_recognizer._eval(X_test, y_test))

# x_train, y_train, x_test, y_test = load()

automl.fit(X_train, y_train, logger=[RichLogger()])
score = automl.score(X_test, y_test)
print(score)