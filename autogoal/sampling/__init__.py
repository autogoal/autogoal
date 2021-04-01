import math
import random
import statistics
import pickle
import numpy as np

from typing import Dict, List, Sequence
import abc

from autogoal.utils import nice_repr


class Sampler:
    """
    Provides methods to obtain random samples with various distributions.

    Can receive a `random_state` to guarantee the same values are obtained
    in two different instantiations.
    """

    def __init__(self, *, random_state: int = None):
        self.rand = random.Random(random_state)

    def choice(self, options, handle=None):
        """
        Returns one of the options.

        ##### Examples

        ```python
        >>> sampler = Sampler(random_state=0)
        >>> [sampler.choice(['A', 'B', 'C']) for _ in range(5)]
        ['B', 'B', 'A', 'B', 'C']

        ```
        """
        return self.categorical(options, handle=handle)

    def distribution(self, name: str, handle=None, **kwargs):
        """
        Shortcut function for generating from a distribution,
        either `discrete`, `continuous`, `boolean` or `categorical`.
        """
        try:
            return getattr(self, name)(handle=handle, **kwargs)
        except AttributeError:
            raise ValueError("Unrecognized distribution name: %s" % name)

    def discrete(self, min=0, max=10, handle=None):
        """
        Returns a discrete value between `min` and `max`.

        ##### Examples

        ```python
        >>> sampler = Sampler(random_state=0)
        >>> [sampler.discrete(0, 10) for _ in range(10)]
        [6, 6, 0, 4, 8, 7, 6, 4, 7, 5]

        ```
        """
        return self.rand.randint(min, max)

    def continuous(self, min=0, max=1, handle=None):
        """
        Returns a continuous value between `min` and `max`.

        ##### Examples

        ```python
        >>> sampler = Sampler(random_state=0)
        >>> [round(sampler.continuous(0, 10), 2) for _ in range(10)]
        [8.44, 7.58, 4.21, 2.59, 5.11, 4.05, 7.84, 3.03, 4.77, 5.83]

        ```
        """
        return self.rand.uniform(min, max)

    def boolean(self, handle=None):
        """
        Returns a boolean value.

        ##### Examples

        ```python
        >>> sampler = Sampler(random_state=0)
        >>> [sampler.boolean() for _ in range(10)]
        [False, False, True, True, False, True, False, True, True, False]

        ```
        """
        return self.rand.uniform(0, 1) < 0.5

    def categorical(self, options, handle=None):
        """
        Returns one of the options.

        The difference between `choice` and `categorical` is evident in more specialized
        classes of `Sampler`. In the default implementation, their behavior is exactly the same.

        ##### Examples

        ```python
        >>> sampler = Sampler(random_state=0)
        >>> [sampler.categorical(['A', 'B', 'C']) for _ in range(5)]
        ['B', 'B', 'A', 'B', 'C']

        ```
        """
        return self.rand.choice(options)


class ModelSampler(Sampler):
    """
    A sampler that builds and uses an internal probabilistic model to generate
    values with a non-uniform probability.

    For the model to work, the `handler` parameter in each sampling method
    must be suplied, otherwise it behaves exactly as the standard `Sampler`.
    """

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

        weights = [
            self._get_model_params(option, UnormalizedWeightParam(value=1))
            for option in options
        ]
        idx = self.rand.choices(
            range(len(options)), weights=[w.value for w in weights], k=1
        )[0]
        option = options[idx]
        self._register_update(option, 1)
        return option

    def discrete(self, min=0, max=10, handle=None):
        if handle is None:
            return super().discrete(min, max, handle)

        params = self._get_model_params(
            handle, MeanDevParam(mean=(min + max) / 2, dev=(max - min))
        )
        value = self._clamp(int(self.rand.gauss(params.mean, params.dev)), min, max)
        return self._register_update(handle, value)

    def continuous(self, min=0, max=1, handle=None):
        if handle is None:
            return super().continuous(min, max, handle)

        params = self._get_model_params(
            handle, MeanDevParam(mean=(min + max) / 2, dev=(max - min))
        )
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

        params = self._get_model_params(
            handle, DistributionParam(weights=[1 for _ in options])
        )
        idx = self.rand.choices(range(len(options)), weights=params.weights, k=1)[0]
        return options[self._register_update(handle, idx)]


class ReplaySampler:
    """
    A sampler that records the generated values and then can replay the
    same outputs in the same order.

    One of the most interesting use cases for `ReplaySampler` is in conjunction with context free
    or graph grammars, for generating complex objects.
    You can pass a sampler wrapped in a `ReplaySampler` during generation, and then
    reuse it later for generating the same object or graph.

    ##### Examples

    First, instantiate a `ReplaySampler` with an internal `Sampler` instance and
    use it normally.

    ```python
    >>> sampler = ReplaySampler(Sampler(random_state=0))
    >>> [sampler.discrete(0,10) for _ in range(10)]
    [6, 6, 0, 4, 8, 7, 6, 4, 7, 5]

    ```

    Then call the `replay` method and reuse the same values.
    `replay()` returns the same instance, to enable chaining method calls.

    ```python
    >>> sampler.replay()
    <autogoal.sampling.ReplaySampler object at ...>
    >>> [sampler.discrete(0,10) for _ in range(5)]
    [6, 6, 0, 4, 8]
    >>> [sampler.discrete(0,10) for _ in range(5)]
    [7, 6, 4, 7, 5]

    ```

    If you try to use it in a different way as originally, it will complain.

    ```python
    >>> sampler.replay().discrete(0,5)
    Traceback (most recent call last):
        ...
    TypeError: Invalid invocation of `discrete` with `args=(0, 5)`, replay history says args='(0, 10)'.

    >>> sampler.replay().boolean()
    Traceback (most recent call last):
        ...
    TypeError: Invalid invocation of `boolean`, replay history says discrete comes next.

    ```
    """

    RECORD = "record"
    REPLAY = "replay"

    def __init__(self, sampler):
        self.sampler = sampler
        self._mode = ReplaySampler.RECORD
        self._history = []
        self._current_history = []

    def _run(self, method, *args, **kwargs):
        if self._mode == ReplaySampler.RECORD:
            result = getattr(self.sampler, method)(*args, **kwargs)
            self._history.append(
                dict(method=method, args=repr(args), kwargs=repr(kwargs), result=result)
            )

            return result

        elif self._mode == ReplaySampler.REPLAY:
            if not self._current_history:
                raise TypeError(
                    f"Invalid invocation of `{method}`, replay history is empty. Maybe you forgot to call `replay`?"
                )

            top = self._current_history[0]

            if top["method"] != method:
                raise TypeError(
                    f"Invalid invocation of `{method}`, "
                    f"replay history says {top['method']} comes next."
                )

            if top["args"] != repr(args):
                raise TypeError(
                    f"Invalid invocation of `{method}` with `args={repr(args)}`, "
                    f"replay history says args={repr(top['args'])}."
                )

            if top["kwargs"] != repr(kwargs):
                raise TypeError(
                    f"Invalid invocation of `{method}` with `kwargs={repr(kwargs)}`, "
                    f"replay history says kwargs={repr(top['kwargs'])}."
                )

            self._current_history.pop(0)
            return top["result"]

    def replay(self) -> "ReplaySampler":
        self._mode = ReplaySampler.REPLAY
        self._current_history = list(self._history)
        return self

    def save(self, fp):
        """
        Saves the state of a `ReplaySampler` to a stream. It must be in replay mode.

        You are responsible for opening and closing the stream yourself.

        ##### Examples

        In this example we create a sampler, and save its state into a `StringIO`
        stream to be able to see what's being saved.

        ```python
        >>> sampler = ReplaySampler(Sampler(random_state=0))
        >>> [sampler.discrete(0, 10) for _ in range(3)]
        [6, 6, 0]

        >>> import io
        >>> fp = io.BytesIO()
        >>> sampler.replay().save(fp)
        >>> len(fp.getvalue())
        183

        ```
        """
        if self._mode != ReplaySampler.REPLAY:
            raise TypeError(
                "A sampler must be in replay mode, i.e., call the `replay()` method."
            )

        pickle.Pickler(fp).dump(self._history)

    @staticmethod
    def load(fp) -> "ReplaySampler":
        """
        Creates a `ReplaySampler` from a stream and returns it already in
        replay mode.

        You are responsible for opening and closing the stream yourself.

        ##### Examples

        ```python
        >>> sampler = ReplaySampler(Sampler(random_state=1))
        >>> [sampler.discrete(0, 10) for _ in range(10)]
        [2, 9, 1, 4, 1, 7, 7, 7, 10, 6]

        >>> import io
        >>> fp = io.BytesIO()
        >>> sampler.replay().save(fp)
        >>> fp.seek(0)
        0
        >>> other_sampler = ReplaySampler.load(fp)
        >>> [other_sampler.discrete(0, 10) for _ in range(5)]
        [2, 9, 1, 4, 1]
        >>> [other_sampler.discrete(0, 10) for _ in range(5)]
        [7, 7, 7, 10, 6]

        """
        history = pickle.Unpickler(fp).load()
        sampler = ReplaySampler(None)
        sampler._history = history
        return sampler.replay()

    def choice(self, *args, **kwargs):
        return self._run("choice", *args, **kwargs)

    def distribution(self, *args, **kwargs):
        return self._run("distribution", *args, **kwargs)

    def discrete(self, *args, **kwargs):
        return self._run("discrete", *args, **kwargs)

    def continuous(self, *args, **kwargs):
        return self._run("continuous", *args, **kwargs)

    def boolean(self, *args, **kwargs):
        return self._run("boolean", *args, **kwargs)

    def categorical(self, *args, **kwargs):
        return self._run("categorical", *args, **kwargs)

    def __getattr__(self, attr):
        if attr == "sampler":
            return self.__dict__.get("sampler")

        return getattr(self.sampler, attr)


class ModelParam(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self, alpha: float, updates) -> "ModelParam":
        pass


@nice_repr
class UnormalizedWeightParam(ModelParam):
    def __init__(self, value):
        self.value = value

    def update(self, alpha: float, updates: list) -> "UnormalizedWeightParam":
        return UnormalizedWeightParam(self.value + alpha * sum(updates))

    def weighted(self, solutions):
        result = 0
        for s, w in solutions:
            result += s * w

        return UnormalizedWeightParam(1 + result)


@nice_repr
class DistributionParam(ModelParam):
    def __init__(self, weights):
        total = sum(weights) or 1
        self.weights = [w / total for w in weights]

    def update(self, alpha: float, updates: list) -> "DistributionParam":
        weights = list(self.weights)

        for i in updates:
            weights[i] += alpha

        return DistributionParam(weights)

    def weighted(self, solutions):
        weights = [1] * len(self.weights)

        for s, w in solutions:
            weights[s] += w

        return DistributionParam(weights)


@nice_repr
class MeanDevParam(ModelParam):
    def __init__(self, mean, dev, *, initial_params=None):
        self.mean = mean
        self.dev = dev

        if initial_params is None:
            self.initial_params = (mean, dev)
        else:
            self.initial_params = initial_params

    def update(self, alpha: float, updates) -> "MeanDevParam":
        new_mean = statistics.mean(updates)
        new_dev = statistics.stdev(updates, new_mean) if len(updates) > 1 else 0

        return MeanDevParam(
            mean=self.mean * (1 - alpha) + new_mean * alpha,
            dev=self.dev * (1 - alpha) + new_dev * alpha,
            initial_params=self.initial_params,
        )

    def weighted(self, solutions):
        values = np.asarray(
            [s for s, w in solutions]
            + [
                self.initial_params[0] - 2 * self.initial_params[1],
                self.initial_params[0] + 2 * self.initial_params[1],
            ]
        )
        weights = np.asarray([w for s, w in solutions] + [1, 1])

        average = np.average(values, weights=weights)
        variance = np.average((values - average) ** 2, weights=weights)

        return MeanDevParam(
            average, math.sqrt(variance), initial_params=self.initial_params
        )


@nice_repr
class WeightParam(ModelParam):
    def __init__(self, value):
        self.value = value

    def update(self, alpha: float, updates) -> "WeightParam":
        new_value = statistics.mean(updates)
        return WeightParam(value=self.value * (1 - alpha) + new_value * alpha)

    def weighted(self, solutions):
        values = np.asarray([s for s, w in solutions] + [0, 1])
        weights = np.asarray([w for s, w in solutions] + [1, 1])

        return WeightParam(np.average(values, weights=weights))


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

    ##### Parameters

    * `values: List`: Values to compare, must be a sortable type (e.g., `int`, `float`, ...).
    * `k: int`: Number of indices to calculate. Defaults to `1`.
    * `maximize: bool`: Whether to compute the maximum or minimum values. Defaults to `False`, i.e., minimize by default.

    ##### Returns:

    * `indices: List[int]`: list of the indices that correspond to maximum (or minimum) values in `values`.

    ##### Examples:

    ```python
    >>> best_indices([.33, 0.12, 0.55, 0.09], k=2)
    [1, 3]

    >>> best_indices([.33, 0.12, 0.55, 0.09], k=3, maximize=True)
    [0, 1, 2]

    >>> best_indices([.33, 0.12, 0.55, 0.09])
    [3]

    ```

    !!! note
        Note that indices are returned in their original order, **not** in the order in which
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

    ##### Parameters:

    * `updates: Sequence[Dict]`: Sequence of update dictionaries obtained
      from calling `ModelSampler.updates`.

    ##### Returns:

    * `update: Dict`: A single dictionary with the combined (appended) updates.

    ##### Examples:

    ```python
    >>> up1 = {'a': [1]}
    >>> up2 = {'b': [2,3]}
    >>> up3 = {'a': [4]}
    >>> merge_updates(up1, up2, up3)
    {'a': [1, 4], 'b': [2, 3]}

    ```
    """
    result = {}

    for upd in updates:
        for key, value in upd.items():
            if not key in result:
                result[key] = []

            result[key].extend(value)

    return result
