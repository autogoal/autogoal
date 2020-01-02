# coding: utf8

import random


class Sampler:
    def __init__(self):
        self.rand = random.Random()

    def choice(self, options):
        return self.rand.choice(options)

    def distribution(self, name: str, **kwargs):
        if name == "discrete":
            return self.rand.randint(kwargs["min"], kwargs["max"])
        elif name == "continuous":
            return self.rand.uniform(kwargs["min"], kwargs["max"])
        elif name == "boolean":
            return self.rand.uniform(0, 1) < 0.5
        elif name == "categorical":
            return self.rand.choice(kwargs["options"])

        raise ValueError("Unrecognized distribution name: %s" % name)


class Grammar:
    def __init__(self, start):
        self._start = start

    def sample(self, *, max_iterations: int = 100, sampler: Sampler = None):
        if sampler is None:
            sampler = Sampler()

        return self._sample(
            symbol=self._start, max_iterations=max_iterations, sampler=sampler
        )

    def _sample(self, symbol, max_iterations, sampler):
        raise NotImplementedError()
