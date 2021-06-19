from autogoal.kb import AlgorithmBase
from autogoal.kb import algorithm

from autogoal.grammar import CategoricalValue, ContinuousValue, DiscreteValue
from autogoal.kb import Sentence, Seq
from autogoal.kb import Supervised ,VectorCategorical, MatrixContinuousDense
from autogoal.utils import nice_repr

import deepmatcher
from os import path

@nice_repr
class SupervisedTextMatcher(AlgorithmBase): # add doc strings
    """
    
    Params:
        -epoch: The number of times each examples is seen
        
    """
    def __init__( # add more params to customize the model, ie to use in deepmatcher.MatchingModel() or self.model.run_train()
            self,
            preprocessor: algorithm(Seq[Sentence], Seq[Sentence]), # fix annotations, what does preprocessor ?
            epoch:DiscreteValue(min=5, max=100)
        ):
        self.preprocessor = preprocessor
        self.epoch = epoch
        self.model = None
        self._mode = "train"

    def train(self):
        self._mode = "train"

    def eval(self):
        self._mode = "eval"

    def run(self,  X:Seq[Sentence], y:Supervised[VectorCategorical]) -> VectorCategorical: # fix annotations
        if self._mode == "train":
            return self._train(X,y)
        elif self._mode == "eval":
            return self._eval(X)

    def _train(self, X, y):
        self.preprocessor.run(X)

        self.model = deepmatcher.MatchingModel()
        train_dataset = deepmatcher.data.MatchingDataset() # build from X
        validation_dataset = deepmatcher.data.MatchingDataset() # build from y
        self.model.run_train(train_dataset=train_dataset, validation_dataset=validation_dataset
            , best_save_path=path.join('./', 'datasets'), epochs=self.epoch)

        return y

    def _eval(self, X):
        ans = []
        for x in X:
            dataset = deepmatcher.data.MatchingDataset() # build from x
            ans.append(self.model.run_eval(dataset=dataset)) # use self.model.run_prediction() instead ?
        return ans
