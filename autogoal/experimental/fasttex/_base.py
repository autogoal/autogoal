from numpy.lib.function_base import select
from autogoal.kb import AlgorithmBase
import fasttext

from autogoal.grammar import CategoricalValue, BooleanValue, ContinuousValue, DiscreteValue
from autogoal.kb import Sentence, Word, FeatureSet, Seq
from autogoal.kb import Supervised ,VectorCategorical
from autogoal.utils import nice_repr
from autogoal.contrib.sklearn._builder import SklearnWrapper
import abc
import numpy as np






class SupervisedTextClassifier(AlgorithmBase):
    
    def __init__ (self, 
                    epoch:DiscreteValue(min=5, max=25),
                    lr:ContinuousValue(0.1,1.0),
                    wordNgrams:DiscreteValue(1,3)
        ):
        self.epoch = epoch
        self.lr = lr
        self.wordNgrams = wordNgrams
        self.model = None
        self._mode = "train"

    def train(self):
        self._mode = "train"

    def eval(self):
        self._mode = "eval"

    def run(self,  X:Seq[Sentence], y:Supervised[VectorCategorical]) -> VectorCategorical :
        if self._mode == "train":
            return self._train(X,y)
        elif self._mode == "eval":
            return self._eval(X)

    def _train(self, X, y):
        with open('test.train','w') as g:
            text = [str(j) +" " +str(i) for i, j in zip(X,y) ]
            g.writelines(text)
            self.model = fasttext.train_supervised(input="test.train",epoch=self.epoch,lr=self.lr,wordNgrams=self.wordNgrams)
        import os
        os.remove("test.train")
        return y

    def _eval(self, X):
        res = []
        for text in X:
            text = text[:-1]
            res.append(self.model.predict(text)[0][0])
        return res


class Text_Descriptor:
    def __init__(self,*labels) -> None:
        self.labels =labels

    def __eq__(self, other):
        if isinstance(other,str):
            return other in self.labels
        if isinstance(other,Text_Descriptor):
            sorted(self.labels) == sorted(other.labels)
        return False
    def __str__(self):
        return " ".join(self.labels)