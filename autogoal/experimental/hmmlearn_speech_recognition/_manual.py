import numpy as np
from scipy.io import wavfile 
from hmmlearn import hmm
from python_speech_features import mfcc
from os import listdir
from autogoal.experimental.hmmlearn_speech_recognition.util import AudioFile
from autogoal.kb import AlgorithmBase, Supervised
from autogoal.contrib.sklearn._builder import SklearnWrapper
from autogoal.grammar import DiscreteValue, ContinuousValue
from autogoal.kb._semantics import Discrete, VectorDiscrete, Word, Seq


class HMMTrainer:
    def __init__(self, n_components, n_iter, covariance_type):
        self.model = hmm.GaussianHMM(n_components=n_components,
                    covariance_type=covariance_type, n_iter=n_iter)

    def fit(self, X):
        self.model.fit(X)

    def score(self, X):
        return self.model.score(X)


class HMMLearnSpeechRecognizer(SklearnWrapper):

    def __init__(
        self, 
        n_components: DiscreteValue(min=1, max=20), 
        n_iter: DiscreteValue(min=500, max=1500),
        covariance_type='diag'):
        
        super().__init__()
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.models = {}
       

    def _preprocess_input(self, X, y):
        new_input = {}
        for audio_file, label in zip(X, y):
            # read the data
            sampling_freq, audio = wavfile.read(audio_file)

            # Obtain the mfcc features
            mfcc_features = mfcc(audio, sampling_freq)

            # store the features according to it's label
            try:
                new_input[label] = np.append(new_input[label], mfcc_features, axis=0)
            except KeyError:
                new_input[label] = mfcc_features

        return new_input


    def run(self, X: Seq[AudioFile], y: Supervised[Seq[Word]]) -> Seq[Word]:
        super().run(X, y)


    def _train(self, X, y):
        """X is a vector of tuples<Float, DiscreteVector> and y is a vector of labels"""
        preprocessed_input = self._preprocess_input(X, y)
        for label in preprocessed_input:
            self.models[label] = HMMTrainer(self.n_components, self.n_iter, self.covariance_type)
            self.models[label].fit(preprocessed_input[label])

    def _eval(self, X, y=None):
        answers = []
        for audio_file, true_label in zip(X,y):
            # read the data
            sampling_freq, audio = wavfile.read(audio_file)

            # Obtain the mfcc features
            mfcc_features = mfcc(audio, sampling_freq)

            # Compute the best score for a given input
            max_score = float('-inf')
            best_label =None
            for label in self.models:
                score = self.models[label].score(mfcc_features)
                if score > max_score:
                    max_score = score
                    best_label = label
            answers.append(best_label)
        return answers


# if __name__ == '__main__':
#     X_train, y_train, X_test, y_test = [], [], [], []
#     for subfolder in listdir('./autogoal/autogoal/contrib/hmmlearn/data'):
#         base_path = f'./autogoal/autogoal/contrib/hmmlearn/data/{subfolder}/'
#         for audio_file in listdir(base_path)[:-1]:
#             file_path = base_path + audio_file
#             X_train.append(file_path)
#             y_train.append(subfolder)
#         test_audio_file = base_path + listdir(base_path)[-1]
#         X_test.append(test_audio_file)
#         y_test.append(subfolder)

#     speech_recognizer = HMMLearnSpeechRecognizer(4, 1000)
#     speech_recognizer._train(X_train, y_train)
#     print(speech_recognizer._eval(X_test, y_test))
