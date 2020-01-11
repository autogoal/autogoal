import random


class Sampler:
    def __init__(self, *, random_state: int = None):
        self.rand = random.Random(random_state)

    def choice(self, options, handle=None):
        return self.categorical(options, handle=handle)

    def distribution(self, name: str, handle=None, **kwargs):
        try:
            return getattr(self, name)(handle=handle, **kwargs)
        except AttributeError:
            raise ValueError("Unrecognized distribution name: %s" % name)

    def discrete(self, min=0, max=10, handle=None):
        return self.rand.randint(min, max)

    def continuous(self, min=0, max=1, handle=None):
        return self.rand.uniform(min, max)

    def boolean(self, handle=None):
        return self.rand.uniform(0, 1) < 0.5

    def categorical(self, options, handle=None):
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

    def __call__(self, sampler: Sampler = None):
        return self.sample(sampler=sampler)

    def _sample(self, symbol, max_iterations, sampler):
        raise NotImplementedError()
