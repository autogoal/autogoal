from abc import abstractmethod
from autogoal.kb import AlgorithmBase

class AugLyWrapper(AlgorithmBase):
    def __init__(self) -> None:
        self._mode = 'train'

    def train(self):
        self._mode = 'train'

    def eval(self):
        self._mode = 'eval'

    def run(self, *args):
        if self._mode == 'train':
            return self._train(*args)
        elif self._mode == 'eval':
            return self._eval(*args) 
        raise ValueError(f'Invalid model mode {self._mode}')

    @abstractmethod
    def _train(self, *args):
        pass

    @abstractmethod
    def _eval(self, *args):
        pass

class AugLyTransformer(AugLyWrapper):
    def _train(self, X, y=None):
        return self.fit_transform(X)

    def _eval(self, X, y=None):
        return self.transform(X)

    @abstractmethod
    def fit_transform(self, X, y=None):
        pass

    @abstractmethod
    def transform(self, X, y=None):
        pass
