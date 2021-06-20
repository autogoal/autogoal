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
        -attr_summarizer: The attribute summarizer. Takes in two word embedding sequences
            and summarizes the information in them to produce two summary vectors as output.
        -classifier: The neural network to perform match / mismatch classification based
            on attribute similarity representations.
        -epoch: Number of training epochs, i.e., number of times to cycle through the entire training set.
        -label_smoothing: The `label_smoothing` parameter to constructor of :class:`~deepmatcher.optim.SoftNLLLoss` criterion.
    """
    def __init__(
            self,
            preprocessor: algorithm(Seq[Sentence], Seq[Sentence]), # fix annotations
            attr_summarizer: CategoricalValue('sif', 'rnn', 'attention', 'hybrid'),
            classifier: CategoricalValue('2-layer-highway', '2-layer-highway-tanh', '3-layer-residual', '3-layer-residual-relu'),
            epoch: DiscreteValue(min=5, max=100),
            label_smoothing: ContinuousValue(min=0.01, max=0.2)
        ):
        self.preprocessor = preprocessor
        self.attr_summarizer = attr_summarizer
        self.classifier = classifier
        self.epoch = epoch
        self.label_smoothing = label_smoothing
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

        self.model = deepmatcher.MatchingModel(attr_summarizer=self.attr_summarizer, classifier=self.classifier)
        train_dataset = deepmatcher.data.MatchingDataset() # build from X
        validation_dataset = deepmatcher.data.MatchingDataset() # build from y
        self.model.run_train(train_dataset=train_dataset, validation_dataset=validation_dataset,
            best_save_path=path.join('./', 'datasets'), epochs=self.epoch, label_smoothing=self.label_smoothing)

        return y

    def _eval(self, X):
        ans = []
        for x in X:
            dataset = deepmatcher.data.MatchingDataset() # build from x
            ans.append(self.model.run_eval(dataset=dataset)) # use self.model.run_prediction() instead ?
        return ans
