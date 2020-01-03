from typing import Mapping
from autogoal.grammar import Sampler
from ._base import SearchAlgorithm


class ModelSampler(Sampler):
    def __init__(self, model: Mapping = None, **kwargs):
        super().__init__(**kwargs)
        self._model = model or {}
        self._updates = {}

    @property
    def model(self):
        return self._model

    @property
    def updates(self):
        return self._updates

    def _get_model_params(self, handle, default):
        if handle in self._model:
            return self._model[handle]
        else:
            self._model[handle] = default
            return default

    def _register_update(self, handle, result):
        if handle not in self._updates:
            self._updates[handle] = []

        self._updates[handle].append(result)
        return result

    def _clamp(self, x, a, b):
        if x < a:
            return a
        if x > b:
            return b
        return x

    def choice(self, options, handle=None):
        if handle is not None:
            return self._sample_categorical(handle, options)

        weights = [self._get_model_params(option, 1) for option in options]
        idx = self.rand.choices(range(len(options)), weights=weights, k=1)[0]
        option = options[idx]
        self._register_update(option, 1)
        return option

    def _sample_discrete(self, handle, min, max):
        if handle is None:
            return super()._sample_discrete(handle, min, max)

        mean, stdev = self._get_model_params(handle, ((min + max) / 2, (max - min)))
        value = self._clamp(round(self.rand.gauss(mean, stdev)), min, max)
        return self._register_update(handle, value)

    def _sample_continuous(self, handle, min, max):
        if handle is None:
            return super()._sample_continuous(handle, min, max)

        mean, stdev = self._get_model_params(handle, ((min + max) / 2, (max - min)))
        value = self._clamp(self.rand.gauss(mean, stdev), min, max)
        return self._register_update(handle, value)

    def _sample_boolean(self, handle):
        if handle is None:
            return super()._sample_boolean(handle)

        p = self._get_model_params(handle, 0.5)
        value = self.rand.uniform(0, 1) < p
        return self._register_update(handle, value)

    def _sample_categorical(self, handle, options):
        if handle is None:
            return super().choice(options, handle)

        weights = self._get_model_params(handle, [1 for _ in options])
        idx = self.rand.choices(range(len(options)), weights=weights, k=1)[0]
        return options[self._register_update(handle, idx)]
