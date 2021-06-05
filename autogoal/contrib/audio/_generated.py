import librosa
from autogoal.kb import MatrixContinuousDense
from autogoal.grammar import DiscreteValue
from autogoal.utils import nice_repr


@nice_repr
class AudioCommandReader():
    '''
    Class to read the first second of wav audio files.
    '''
    def __init__(self, sample_rate = 8000):
        self._sample_rate = sample_rate

    def run(self, audio_file):
        audio_signal, _ = librosa.load(audio_file, sr=self._sample_rate)
        return audio_signal[:self._sample_rate]


@nice_repr
class AudioVectorToSequence():
    def run(self, audio_signal): 
        return audio_signal.reshape(-1, 1)


@nice_repr
class MelFrequencyCepstralCoefficients():
    '''
    This class receive a vector representing an audio and will
    extract the MFCC as features.
    '''
    def __init__(
        self, 
        sample_rate,
        n_mfcc: DiscreteValue(20, 40),

    ):
        self._n_mfcc = n_mfcc
        self._sample_rate = sample_rate 
    
    def run(self, audio_signal):
        return librosa.feature.mfcc(
            audio_signal,
            sr=self._sample_rate,
            n_mfcc=self._n_mfcc
        )