import scipy

from typing import Mapping, Optional, Dict, List, Sequence
from autogoal.grammar import Sampler
from ._base import SearchAlgorithm


class ModelSampler(Sampler):
    def __init__(self, model: Dict = None, **kwargs):
        super().__init__(**kwargs)
        self._model: Dict = model or {}
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

        weights = [self._get_model_params(option, 1) for option in options]
        idx = self.rand.choices(range(len(options)), weights=weights, k=1)[0]
        option = options[idx]
        self._register_update(option, 1)
        return option

    def discrete(self, min=0, max=10, handle=None):
        if handle is None:
            return super().discrete(min, max, handle)

        mean, stdev = self._get_model_params(handle, ((min + max) / 2, (max - min)))
        value = self._clamp(int(self.rand.gauss(mean, stdev)), min, max)
        return self._register_update(handle, value)

    def continuous(self, min=0, max=1, handle=None):
        if handle is None:
            return super().continuous(min, max, handle)

        mean, stdev = self._get_model_params(handle, ((min + max) / 2, (max - min)))
        value = self._clamp(self.rand.gauss(mean, stdev), min, max)
        return self._register_update(handle, value)

    def boolean(self, handle=None):
        if handle is None:
            return super().boolean(handle)

        p = self._get_model_params(handle, (0.5,))[0]
        value = self.rand.uniform(0, 1) < p
        return self._register_update(handle, value)

    def categorical(self, options, handle=None):
        if handle is None:
            return super().categorical(options, handle)

        weights = self._get_model_params(handle, [1 for _ in options])
        idx = self.rand.choices(range(len(options)), weights=weights, k=1)[0]
        return options[self._register_update(handle, idx)]


def update_model(model, updates, alpha: float = 1):
    new_model = {}

    for handle, params in model.items():
        upd = updates.get(handle)

        if upd is None:
            new_model[handle] = params
            continue

        # TODO: refactor to a more Object Oriented way
        if isinstance(params, (float, int)):
            # float or int means a single un-normalized weight
            new_model[handle] = params + alpha * sum(upd)
        elif isinstance(params, list):
            # a list means a (potentially un-normalized) distribution over categories
            new_model[handle] = list(params)
            for i in upd:
                new_model[handle][i] += alpha
        elif isinstance(params, tuple):
            # a tuple means specific distribution parameters, like mean and stdev
            if len(params) == 2:
                mean, stdev = params
                new_mean = scipy.mean(upd)
                new_stdev = scipy.std(upd)
                new_model[handle] = (
                    mean * (1 - alpha) + new_mean * alpha,
                    stdev * (1 - alpha) + new_stdev * alpha,
                )
            elif len(params) == 1:
                p = params[0]
                new_p = upd.count(True)
                new_model[handle] = (p * (1 - alpha) + new_p * alpha,)
            else:
                raise ValueError("Unrecognized params %r" % params)
        else:
            raise ValueError("Unrecognized params %r" % params)

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
        pop_size: int = 100,
        learning_factor: float = 0.05,
        selection: float = 0.2,
        random_state: Optional[int] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._pop_size = pop_size
        self._learning_factor = learning_factor
        self._selection = selection
        self._model: Dict = {}

    def _run_one_generation(self):
        self._samplers = []

        for _ in range(self._pop_size):
            sampler = ModelSampler(self._model)
            self._samplers.append(sampler)
            yield self._generator_fn(sampler=sampler)

    def _finish_generation(self, fns):
        # Compute the marginal model of the best pipelines
        samplers: List[ModelSampler] = [self._samplers[i] for i in best_indices(fns)]
        updates: Dict = merge_updates(*[sampler.updates for sampler in samplers])

        # Update the probabilistic model with the marginal model from the best pipelines
        self._model = update_model(self._model, updates, self._learning_factor)
