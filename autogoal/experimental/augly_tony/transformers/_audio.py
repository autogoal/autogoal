from augly.audio.transforms import BaseTransform
import augly.audio.transforms as transforms

from autogoal.experimental.augly_tony.semantic import Audio
from autogoal.utils import nice_repr
from autogoal.grammar import (
    CategoricalValue,
    DiscreteValue,
    ContinuousValue,
    BooleanValue,
)

import numpy as np

from _util import AugLyTransformer


@nice_repr
class AugLyAudioTransformer(AugLyTransformer):
    """
    Base class for augLy audio transformers
    """

    def __init__(self):
        super().__init__()
        self._transformer: BaseTransform = None

    def get_transformer(self) -> BaseTransform:
        pass

    def run(self, X: Audio) -> Audio:
        if self._transformer is None:
            self._transformer = self.get_transformer()

        return self._transformer(Audio)


max_db_val = 110

# TODO: Allow to add a background audio
@nice_repr
class AddBackgroundNoiseTransformer(AugLyAudioTransformer):
    """
    Mixes in a background sound into the audio
    """

    def __init__(
        self,
        snr_level_db: ContinuousValue(0, max_db_val),
    ):
        super().__init__()
        self.snr_level_db = snr_level_db

    def get_transformer(self) -> BaseTransform:
        return transforms.AddBackgroundNoise(
            snr_level_db=self.snr_level_db,
        )


@nice_repr
class ChangeVolumeTransformer(AugLyAudioTransformer):
    """
    Changes the volume of the audio
    """

    def __init__(
        self,
        volume_db: ContinuousValue(0, max_db_val),
    ):
        super().__init__()
        self.volume_db = volume_db

    def get_transformer(self) -> BaseTransform:
        return transforms.ChangeVolume(
            volume_db=self.volume_db,
        )


@nice_repr
class ClickseTransformer(AugLyAudioTransformer):
    """
    Adds clicks to the audio at a given regular interval
    """

    def __init__(
        self,
        seconds_between_clicks: ContinuousValue(0.01, 5),
    ):
        super().__init__()
        self.seconds_between_clicks = seconds_between_clicks

    def get_transformer(self) -> BaseTransform:
        return transforms.Clicks(
            seconds_between_clicks=self.seconds_between_clicks,
        )


@nice_repr
class ClipTransformer(AugLyAudioTransformer):
    """
    Clips the audio using the specified offset and duration factors
    """

    def __init__(
        self,
        offset_factor: ContinuousValue(0, 1),
        duration_factor: ContinuousValue(0, 1),
    ):
        super().__init__()
        self.offset_factor = offset_factor
        self.duration_factor = duration_factor

    def get_transformer(self) -> BaseTransform:
        return transforms.Clip(
            duration_factor=self.duration_factor,
        )


@nice_repr
class HarmonicTransformer(AugLyAudioTransformer):
    """
    Extracts the harmonic part of the audio
    """

    def __init__(
        self,
        kernel_size: ContinuousValue(0, 31),
        power: ContinuousValue(0, 2),
        margin: ContinuousValue(0, 1),
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.power = power
        self.margin = margin

    def get_transformer(self) -> BaseTransform:
        return transforms.Harmonic(
            kernel_size=self.kernel_size,
            power=self.power,
            margin=self.margin,
        )


@nice_repr
class PercussiveTransformer(AugLyAudioTransformer):
    """
    Extracts the percussive part of the audio
    """

    def __init__(
        self,
        kernel_size: ContinuousValue(0, 31),
        power: ContinuousValue(0, 2),
        margin: ContinuousValue(0, 1),
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.power = power
        self.margin = margin

    def get_transformer(self) -> BaseTransform:
        return transforms.Percussive(
            kernel_size=self.kernel_size,
            power=self.power,
            margin=self.margin,
        )


@nice_repr
class HighPassFilterTransformer(AugLyAudioTransformer):
    """
    Allows audio signals with a frequency higher than the given cutoff to pass
    through and attenuates signals with frequencies lower than the cutoff frequency
    """

    def __init__(
        self,
        cutoff_hz: ContinuousValue(0, 3000.0),
    ):
        super().__init__()
        self.cutoff_hz = cutoff_hz

    def get_transformer(self) -> BaseTransform:
        return transforms.HighPassFilter(
            cutoff_hz=self.cutoff_hz,
        )


@nice_repr
class LowPassFilterTransformer(AugLyAudioTransformer):
    """
    Allows audio signals with a frequency higher than the given cutoff to pass
    through and attenuates signals with frequencies lower than the cutoff frequency
    """

    def __init__(
        self,
        cutoff_hz: ContinuousValue(0, 3000.0),
    ):
        super().__init__()
        self.cutoff_hz = cutoff_hz

    def get_transformer(self) -> BaseTransform:
        return transforms.LowPassFilter(
            cutoff_hz=self.cutoff_hz,
        )


# TODO: Allow add a custom background audio
@nice_repr
class InsertInBackgroundTransformer(AugLyAudioTransformer):
    """
    Mixes in a background sound into the audio
    """

    def __init__(
        self,
        offset_factor: ContinuousValue(0, 1),
    ):
        super().__init__()
        self.offset_factor = offset_factor

    def get_transformer(self) -> BaseTransform:
        return transforms.InsertInBackground(
            offset_factor=self.offset_factor,
        )


@nice_repr
class InvertChannelsTransformer(AugLyAudioTransformer):
    """
    Inverts the channels of the audio.
    """

    def get_transformer(self) -> BaseTransform:
        return transforms.InvertChannels()


@nice_repr
class NormalizeTransformer(AugLyAudioTransformer):
    """
    Normalizes the audio array along the chosen axis (norm(audio, axis=axis) == 1)
    """

    def __init__(
        self,
        norm: CategoricalValue(np.inf, -np.inf, 0),
        axis: DiscreteValue(0, 4),
        threshold: ContinuousValue(0, max_db_val),
        fill: BooleanValue(),
    ):
        super().__init__()
        self.norm = norm
        self.axis = axis
        self.threshold = threshold
        self.fill = fill

    def get_transformer(self) -> BaseTransform:
        return transforms.Normalize(
            norm=self.norm,
            axis=self.axis,
            threshold=self.threshold,
            fill=self.fill,
        )


@nice_repr
class PitchShiftTransformer(AugLyAudioTransformer):
    """
    Shifts the pitch of the audio by `n_steps`
    """

    def __init__(
        self,
        n_steps: ContinuousValue(0, 100),
    ):
        super().__init__()
        self.n_steps = n_steps

    def get_transformer(self) -> BaseTransform:
        return transforms.PitchShift(
            n_steps=self.n_steps,
        )


@nice_repr
class SpeedTransformer(AugLyAudioTransformer):
    """
    Changes the speed of the audio, affecting pitch as well
    """

    def __init__(
        self,
        factor: ContinuousValue(0, 4),
    ):
        super().__init__()
        self.factor = factor

    def get_transformer(self) -> BaseTransform:
        return transforms.Speed(
            factor=self.factor,
        )


@nice_repr
class TempoTransformer(AugLyAudioTransformer):
    """
    Adjusts the tempo of the audio by a given factor (without
    affecting the pitch)
    """

    def __init__(
        self,
        factor: ContinuousValue(0, 4),
    ):
        super().__init__()
        self.factor = factor

    def get_transformer(self) -> BaseTransform:
        return transforms.Tempo(
            factor=self.factor,
        )


@nice_repr
class TimeStretchTransformer(AugLyAudioTransformer):
    """
    Time-stretches the audio by a fixed rate
    """

    def __init__(
        self,
        rate: ContinuousValue(0, 4),
    ):
        super().__init__()
        self.rate = rate

    def get_transformer(self) -> BaseTransform:
        return transforms.TimeStretch(
            rate=self.rate,
        )


@nice_repr
class ToMonoTransformer(AugLyAudioTransformer):
    """
    Converts the audio from stereo to mono by averaging samples across channels
    """

    def get_transformer(self) -> BaseTransform:
        return transforms.ToMono()


__all__ = [
AddBackgroundNoiseTransformer,
ChangeVolumeTransformer,
ClickseTransformer,
ClipTransformer,
HarmonicTransformer,
PercussiveTransformer,
HighPassFilterTransformer,
LowPassFilterTransformer,
InsertInBackgroundTransformer,
InvertChannelsTransformer,
NormalizeTransformer,
PitchShiftTransformer,
SpeedTransformer,
TempoTransformer,
TimeStretchTransformer,
ToMonoTransformer,
]