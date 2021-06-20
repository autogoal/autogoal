from os import mkdir, remove
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
import random
import string

CACHE = Path(__file__).parent / '.cache'

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

def random_string():
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(10))

def split_X(X):
    idx = 0
    for row in X:
        if row[0] == None:
            break
        idx += 1
    
    if idx+1 < len(X):
        vX = X[idx+1:]
        X = X[:idx]
    else:
        vX = None
    return X, vX

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
        X = [row.copy() for row in X]
        for i in range(len(X)):
            X[i].insert(1, y[i])

        X, vX = split_X(X)
        assert vX

        if not Path.exists(CACHE):
            mkdir(CACHE)

        train = CACHE / f'{random_string()}.train'
        with open(train, 'w') as f:
            w = csv.writer(f)
            w.writerows(X)
        
        val = CACHE / f'{random_string()}.val'
        with open(val, 'w') as f:
            vw = csv.writer(f)
            vw.writerows(vX)

        self.model = deepmatcher.MatchingModel(attr_summarizer=self.attr_summarizer, classifier=self.classifier)
        train_dataset = buildMatchingSet(train)
        validation_dataset = buildMatchingSet(val)
        self.model.run_train(train_dataset=train_dataset, validation_dataset=validation_dataset,
            best_save_path=CACHE/f'best_model{random_string()}.pth', epochs=self.epoch, label_smoothing=self.label_smoothing)

        remove(train)
        remove(val)
        return y

    def _eval(self, X):
        X = [row.copy() for row in X]
        # ans = []
        try:
            X, vX = split_X(X)
            # X += vX[1:]
        except:
            pass

        X[0].insert(1, 'label') # need to switch to process_unlabeled
        for i in range(1, len(X)):
            X[i].insert(1, 0)

        eval = CACHE / f'{random_string()}.eval'
        with open(eval, 'w') as f:
            w = csv.writer(f)
            w.writerows(X)

        eval_dataset = buildMatchingSet(eval)
        # eval_dataset = deepmatcher.data.process_unlabeled(path=CACHE / 'temp.val',
        #     trained_model=self.model, ignore_columns=('left_id', 'right_id'))
        
        # with open(CACHE / 'temp.eval.txt', 'w') as f: # just to see logs
        #     self.model.run_prediction(eval_dataset, True).to_csv(f)
        
        remove(eval)
        return [1 if x >= 0.5 else 0 for x in self.model.run_prediction(eval_dataset).to_dict()['match_score'].values()]
        # for x in X:
        #     dataset = deepmatcher.data.MatchingDataset() # build from x
        #     ans.append(self.model.run_eval(dataset=dataset)) # use self.model.run_prediction() instead ?
        # return ans

if __name__ == '__main__':
    s = SupervisedTextMatcher(
        attr_summarizer='attention',
        classifier='2-layer-highway',
        epoch=2,
        label_smoothing=0.02
    )
    test_name = 'Fodors-Zagats'
    from autogoal.experimental.deepmatcher.dataset import DeepMatcherDataset
    from autogoal.experimental.deepmatcher import DATASETS
    X_train, y_train , X_test , y_test = DeepMatcherDataset(test_name, DATASETS[test_name]).load()
    s.train()
    s.run(X_train, y_train)
    s.eval()
    print(s.run(X_test, y_test)[:10])
