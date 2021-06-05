import numpy as np
from ._generated import *
from autogoal.experimental.audio_command_recognition.kb._semantics import AudioFile, AudioFeatures
from autogoal.kb import *
from autogoal.grammar import DiscreteValue, CategoricalValue
from autogoal.kb import algorithm
from autogoal.utils import nice_repr


@nice_repr
class AudioCommandPreprocessor(AlgorithmBase):
    '''
    This class can read audio files, assuming they are audio commands with 1 second length,
    and return audio as sequences, each unit of the sequence will be a features vector depending
    on the algorithm selected as preprocessor.
    '''
    def __init__(
        self, 
        n_mfcc : DiscreteValue(20, 40) = 20,
        preprocessor: CategoricalValue("AudioVectorToSequence", "MelFrequencyCepstralCoefficients") = "AudioVectorToSequence"
    ):
        self._sample_rate = 8000
        self._n_mfcc = n_mfcc
        self._audio_reader = AudioCommandReader()
        self._preprocessor = self._build_preprocessor(preprocessor)
    
    def _build_preprocessor(self, preprocessor_type):
        preprocessor = None
        
        if preprocessor_type == "AudioVectorToSequence":
            preprocessor = AudioVectorToSequence()

        elif preprocessor_type == "MelFrequencyCepstralCoefficients":
            preprocessor = MelFrequencyCepstralCoefficients(self._sample_rate, self._n_mfcc)
    
        return preprocessor

    def run(self, audio_file: AudioFile) -> AudioFeatures:
        audio_signal = self._audio_reader.run(audio_file)

        return self._preprocessor.run(audio_signal)
        

@nice_repr
class AudioClassifier(AlgorithmBase):
    '''
    This class is Classifier able to receive a bunch of wav files and classify them
    with specific labels
    '''
    def __init__(
        self,
        audio_preprocessor: algorithm(AudioFile, AudioFeatures),
        classifier: algorithm(AudioFeatures, Supervised[VectorCategorical], VectorCategorical),
    ):
        self._audio_preprocessor = audio_preprocessor
        self._classifier = classifier
        self._mode = "train"
    
    def train(self):
        self._mode = "train"
        self._classifier.train()

    def eval(self):
        self._mode = "eval"
        self._classifier.eval()

    def fit(self, audio_files, labels):
        processed_audios = []

        for audio_file in audio_files:
            processed_audios.append(self._audio_preprocessor.run(audio_file))
        processed_audios = np.array(processed_audios) 
        self._classifier.fit(processed_audios, labels) 
        
    def predict(self, audio_files):
        processed_audios = []
        for audio_file in audio_files:
            processed_audios.append(self._audio_preprocessor.run(audio_file))
        
        processed_audios = np.array(processed_audios) 
        return self._classifier.predict(processed_audios)

    def run(self, data: Seq[AudioFile], labels : Supervised[Categorical]) -> VectorCategorical:
        if self._mode == "train":
            self.fit(data, labels)
            return labels
        
        if self._mode == "eval":
            return self.predict(data)