# TODO move to sampler folder when out of experimental
from typing import Dict
from autogoal.sampling import Sampler


class ExactSampler(Sampler):
    """
    A sampler that builds and uses an internal state to generate
    a solution with given values.

    For the model to work, the `handler` parameter in each sampling method
    must exist within the intenal model state, or it will thrown an error.
    """

    def __init__(self, model: Dict = None, **kwargs):
        super().__init__(**kwargs)
        self._model: Dict = {} if model is None else model

    @property
    def model(self):
        return self._model

    def _get_model_params(self, handle):
        if handle in self._model:
            return self._model[handle]
        else:
            raise ValueError("Incomplete exact sampler model")

    def _get_algorithm_options_params(self, options):
        chosen = None
        for option in options:
            if option in self._model:
                if chosen is not None:
                    raise ValueError(
                        f"Ambiguous model algorithm values {chosen} and {option}"
                    )
                chosen = option
        if chosen is None:
            raise ValueError("Incomplete exact sampler model.")
        return chosen

    def choice(self, options, handle=None):
        param = None
        if handle is None:
            param = self._get_algorithm_options_params(options)
        else:
            param = self._get_model_params(handle)
        if param in options:
            return param
        else:
            raise ValueError(
                f"Exact sampler model with incorrect choice parameter {handle}={param}"
            )

    def discrete(self, min=0, max=10, handle=None):
        param = self._get_model_params(handle)
        if int(param) >= min and int(param) <= max:
            return param
        else:
            raise ValueError(
                f"Exact sampler model with incorrect discrete parameter {handle}={param}"
            )

    def continuous(self, min=0, max=1, handle=None):
        param = self._get_model_params(handle)
        if param >= min and param <= max:
            return param
        else:
            raise ValueError(
                f"Exact sampler model with incorrect continuous parameter {handle}={param}"
            )

    def boolean(self, handle=None):
        param = self._get_model_params(handle)
        if param is True or param is False:
            return param
        else:
            raise ValueError(
                f"Exact sampler model with incorrect boolean parameter {handle}={param}"
            )

    def categorical(self, options, handle=None):
        param = self._get_model_params(handle)
        if param in options:
            return param
        else:
            raise ValueError(
                f"Exact sampler model with incorrect categorical parameter {handle}={param}"
            )
