from os import mkdir
from autogoal.kb import AlgorithmBase
from autogoal.kb import algorithm

from autogoal.grammar import CategoricalValue, ContinuousValue, DiscreteValue
from autogoal.kb import Sentence, Seq, Text
from autogoal.kb import Supervised ,VectorCategorical, MatrixContinuousDense
from autogoal.utils import nice_repr

import deepmatcher
from pathlib import Path
import csv
import pandas

CACHE = Path(__file__).parent / 'cache'

def buildMatchingSet(file_name):
    return deepmatcher.data.process(
        path=CACHE,
        train=file_name,
        ignore_columns=('left_id', 'right_id'),
        left_prefix='left_',
        right_prefix='right_',
        label_attr='label',
        id_attr='id',
        embeddings='glove.twitter.27B.25d',
        embeddings_cache_path=(CACHE.parent / '.vector_cache')
    )

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
            attr_summarizer: CategoricalValue('sif', 'rnn', 'attention', 'hybrid'),
            classifier: CategoricalValue('2-layer-highway', '2-layer-highway-tanh', '3-layer-residual', '3-layer-residual-relu'),
            epoch: DiscreteValue(min=5, max=100),
            label_smoothing: ContinuousValue(min=0.01, max=0.2)
        ):
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

    def run(self,  X:Seq[Seq[Text]], y:Supervised[VectorCategorical]) -> VectorCategorical: # fix annotations
        if self._mode == "train":
            return self._train(X,y)
        elif self._mode == "eval":
            return self._eval(X)

    def _train(self, X, y):
        for i in range(len(X)):
            X[i].insert(1, y[i])
        idx = 0
        for row in X:
            if row[0][0] == 'v':
                break
            idx += 1
        
        assert idx < len(X)
        vX = X[idx:]
        X = X[:idx]
        for i in range(len(vX)):
            vX[i][0] = vX[i][0][1:]

        if not Path.exists(CACHE):
            mkdir(CACHE)

        with open(CACHE / 'temp.train', 'w') as f:
            w = csv.writer(f)
            w.writerows(X)
        with open(CACHE / 'temp.val', 'w') as f:
            vw = csv.writer(f)
            vw.writerows(vX)

        self.model = deepmatcher.MatchingModel(attr_summarizer=self.attr_summarizer, classifier=self.classifier)
        train_dataset = buildMatchingSet('temp.train')
        validation_dataset = buildMatchingSet('temp.val')
        self.model.run_train(train_dataset=train_dataset, validation_dataset=validation_dataset,
            best_save_path=CACHE, epochs=self.epoch, label_smoothing=self.label_smoothing)

        return y

    def _eval(self, X):
        # ans = []
        w = csv.writer(CACHE / 'temp.eval')
        w.writerows(X)
        eval_dataset = buildMatchingSet('temp.eval')
        return self.model.run_prediction(eval_dataset).to_dict()['match_score'].values()
        # for x in X:
        #     dataset = deepmatcher.data.MatchingDataset() # build from x
        #     ans.append(self.model.run_eval(dataset=dataset)) # use self.model.run_prediction() instead ?
        # return ans

if __name__ == '__main__':
    s = SupervisedTextMatcher(
        attr_summarizer='sif',
        classifier='2-layer-highway',
        epoch=50,
        label_smoothing=0.02
    )
    test_name = 'Fodors-Zagats'
    from autogoal.experimental.deepmatcher.dataset import DeepMatcherDataset
    from autogoal.experimental.deepmatcher import DATASETS
    X_train, y_train , X_test , y_test = DeepMatcherDataset(test_name, DATASETS[test_name]).load()
    s.train()
    s.run(X_train, y_train)
