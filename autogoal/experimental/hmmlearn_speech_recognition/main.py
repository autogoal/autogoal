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


class HMMLearnSpeechRecognizer(AlgorithmBase):

    def __init__(
        self, 
        n_components: DiscreteValue(min=1, max=20), 
        n_iter: DiscreteValue(min=500, max=1500),
        covariance_type='diag'):        
        self._mode = 'train'
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.models = {}
        super().__init__()
       

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

    def train(self):
        self._mode = 'train'
    
    def eval(self):
        self._mode = 'eval'


    def run(self, X: Seq[AudioFile], y: Supervised[Seq[Word]]) -> Seq[Word]:
        if (self._mode == 'train'):
            self._train(X, y)
            return y
        else:
            return self._eval(X)

    def _train(self, X, y):
        preprocessed_input = self._preprocess_input(X, y)
        for label in preprocessed_input:
            self.models[label] = HMMTrainer(self.n_components, self.n_iter, self.covariance_type)
            self.models[label].fit(preprocessed_input[label])
        return y

    def _eval(self, X):
        answers = []
        for audio_file in X:
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
