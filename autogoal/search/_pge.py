import statistics
import abc

from typing import Mapping, Optional, Dict, List, Sequence
from autogoal.grammar import Sampler
from ._base import SearchAlgorithm


class ModelSampler(Sampler):
    def __init__(self, model: Dict = None, **kwargs):
        super().__init__(**kwargs)
        self._model: Dict = {} if model is None else model
        self._updates: Dict = {}

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
            return self.categorical(options, handle)

        weights = [self._get_model_params(option, UnormalizedWeightParam(value=1)) for option in options]
        idx = self.rand.choices(range(len(options)), weights=[w.value for w in weights], k=1)[0]
        option = options[idx]
        self._register_update(option, 1)
        return option

    def discrete(self, min=0, max=10, handle=None):
        if handle is None:
            return super().discrete(min, max, handle)

        params = self._get_model_params(handle, MeanDevParam(mean=(min + max) / 2, dev=(max - min)))
        value = self._clamp(int(self.rand.gauss(params.mean, params.stdev)), min, max)
        return self._register_update(handle, value)

    def continuous(self, min=0, max=1, handle=None):
        if handle is None:
            return super().continuous(min, max, handle)

        params = self._get_model_params(handle, MeanDevParam(mean=(min + max) / 2, dev=(max - min)))
        value = self._clamp(self.rand.gauss(params.mean, params.dev), min, max)
        return self._register_update(handle, value)

    def boolean(self, handle=None):
        if handle is None:
            return super().boolean(handle)

        params = self._get_model_params(handle, WeightParam(value=0.5))
        value = self.rand.uniform(0, 1) < params.value
        return self._register_update(handle, value)

    def categorical(self, options, handle=None):
        if handle is None:
            return super().categorical(options, handle)

        params = self._get_model_params(handle, DistributionParam(weights=[1 for _ in options]))
        idx = self.rand.choices(range(len(options)), weights=params.weights, k=1)[0]
        return options[self._register_update(handle, idx)]


class ModelParam(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self, alpha: float, updates) -> "ModelParam":
        pass


class UnormalizedWeightParam(ModelParam):
    def __init__(self, value):
        self.value = value

    def update(self, alpha: float, updates: list) -> "UnormalizedWeightParam":
        return UnormalizedWeightParam(self.value + alpha * sum(updates))


class DistributionParam(ModelParam):
    def __init__(self, weights):
        total = sum(weights) or 1
        self.weights = [w / total for w in weights]

    def update(self, alpha: float, updates: list) -> "DistributionParam":
        weights = list(self.weights)

        for i in updates:
            weights[i] += alpha

        return DistributionParam(weights)


class MeanDevParam(ModelParam):
    def __init__(self, mean, dev):
        self.mean = mean
        self.dev = dev

    def update(self, alpha: float, updates) -> "MeanDevParam":
        new_mean = statistics.mean(updates)
        new_dev = statistics.stdev(updates, new_mean) if len(updates) > 1 else 0

        return MeanDevParam(
            mean=self.mean * (1 - alpha) + new_mean * alpha,
            dev=self.dev * (1 - alpha) + new_dev * alpha,
        )


class WeightParam(ModelParam):
    def __init__(self, value):
        self.value = value

    def update(self, alpha: float, updates) -> "WeightParam":
        new_value = statistics.mean(updates)
        return WeightParam(value=self.value * (1 - alpha) + new_value * alpha,)


def update_model(model, updates, alpha: float = 1):
    new_model = {}

    for handle, params in model.items():
        upd = updates.get(handle)

        if upd is None:
            new_model[handle] = params
        else:
            new_model[handle] = params.update(alpha, upd)

    return new_model


def _argsort(l):
    # taken from https://stackoverflow.com/questions/6422700
    return sorted(range(len(l)), key=l.__getitem__)


def best_indices(values: List, k: int = 1, maximize: bool = False) -> List[int]:
    """
    Computes the `k` best indices from values, i.e., the indices of the values
    that are the top minimum (or maximum).

    Args:

    * `values: List`: Values to compare, must be a sortable type (e.g., `int`, `float`, ...).
    * `k: int`: Number of indices to calculate. Defaults to `1`.
    * `maximize: bool`: Whether to compute the maximum or minimum values. Defaults to `False`, i.e., minimize by default.

    Returns:

    * `indices: List[int]`: list of the indices that correspond to maximum (or minimum) values in `values`.

    Examples:

        >>> best_indices([.33, 0.12, 0.55, 0.09], k=2)
        [1, 3]

        >>> best_indices([.33, 0.12, 0.55, 0.09], k=3, maximize=True)
        [0, 1, 2]

        >>> best_indices([.33, 0.12, 0.55, 0.09])
        [3]

    !!! note
        Note that indices are returned in sorted index order, **not** in the order in which
        the values would be sorted themselves.
    """
    indices = _argsort(_argsort(values))

    if maximize:
        threshold = len(values) - k
        return [i for i in range(len(values)) if indices[i] >= threshold]
    else:
        threshold = k
        return [i for i in range(len(values)) if indices[i] < threshold]


def merge_updates(*updates: Sequence[Dict]) -> Dict:
    """
    Merges a bunch of update dicts from `ModelSampler`
    into a single dictionary.

    Args:

    * `updates: Sequence[Dict]`: Sequence of update dictionaries obtained
      from calling `ModelSampler.updates`.

    Returns:

    * `update: Dict`: A single dictionary with the combined (appended) updates.

    Examples:

        >>> up1 = {'a': [1]}
        >>> up2 = {'b': [2,3]}
        >>> up3 = {'a': [4]}
        >>> merge_updates(up1, up2, up3)
        {'a': [1, 4], 'b': [2, 3]}

    """
    result = {}

    for upd in updates:
        for key, value in upd.items():
            if not key in result:
                result[key] = []

            result[key].extend(value)

    return result


class PESearch(SearchAlgorithm):
    def __init__(
        self,
        *args,
        learning_factor: float = 0.05,
        selection: float = 0.2,
        epsilon_greed: float = 0.1,
        random_state: Optional[int] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._learning_factor = learning_factor
        self._selection = selection
        self._epsilon_greed = epsilon_greed
        self._model: Dict = {}

    def _start_generation(self):
        self._samplers = []

    def _build_sampler(self):
        if len(self._samplers) < self._epsilon_greed * self._pop_size:
            sampler = ModelSampler()
        else:
            sampler = ModelSampler(self._model)

        self._samplers.append(sampler)
        return sampler

    def _finish_generation(self, fns):
        # Compute the marginal model of the best pipelines
        indices = best_indices(
            fns, k=int(self._selection * len(fns)), maximize=self._maximize
        )
        samplers: List[ModelSampler] = [self._samplers[i] for i in indices]
        updates: Dict = merge_updates(*[sampler.updates for sampler in samplers])

        # Update the probabilistic model with the marginal model from the best pipelines
        self._model = update_model(self._model, updates, self._learning_factor)
