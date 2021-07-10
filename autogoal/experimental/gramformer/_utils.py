from abc import abstractmethod
from autogoal.kb import AlgorithmBase

class BaseTransformer(AlgorithmBase):
    def __init__(self):
        self._mode = 'train'

    def train(self):
        self._mode = 'train'
    
    def eval(self):
        self._mode = 'eval'

    def run(self, *args):
        if self._mode == 'train':
            return self.fit_transform(self, *args)
        elif self._mode == 'eval':
            return self.transform(self, *args)
        else:
            raise ValueError(f'Invalid transformer mode')

    @abstractmethod
    def fit_transform(self, *args):
        pass

    @abstractmethod
    def transform(self, *args):
        pass 