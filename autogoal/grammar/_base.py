import random


class Sampler:
    def __init__(self, *, random_state: int = None):
        self.rand = random.Random(random_state)

    def choice(self, options, handle=None):
        return self.rand.choice(options)

    def distribution(self, name: str, handle=None, **kwargs):
        try:
            return getattr(self, "_sample_%s" % name)(handle, **kwargs)
        except AttributeError:
            raise ValueError("Unrecognized distribution name: %s" % name)

    def _sample_discrete(self, handle, min, max):
        return self.rand.randint(min, max)

    def _sample_continuous(self, handle, min, max):
        return self.rand.uniform(min, max)

    def _sample_boolean(self, handle):
        return self.rand.uniform(0, 1) < 0.5

    def _sample_categorical(self, handle, options):
        return self.rand.choice(options)


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
