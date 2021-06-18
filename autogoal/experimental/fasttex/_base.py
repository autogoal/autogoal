from numpy.lib.function_base import select
from autogoal.kb import AlgorithmBase
import fasttext

from autogoal.grammar import CategoricalValue, BooleanValue, ContinuousValue, DiscreteValue
from autogoal.kb import Sentence, Word, FeatureSet, Seq
from autogoal.kb import Supervised ,VectorCategorical, VectorContinuous, MatrixContinuousDense
from autogoal.utils import nice_repr
from autogoal.contrib.sklearn._builder import SklearnTransformer, SklearnWrapper
import abc
import numpy as np

from os import remove
#from time import time_ns

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
        remove("test.train")
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

class UnsupervisedWordRepresentation(AlgorithmBase):   
    def __init__ (self,
                    min_subword: DiscreteValue(min=1, max=3),
                    max_subword: DiscreteValue(min=3, max=6),
                    dimension: DiscreteValue(min=100, max=300),
                    model: CategoricalValue("skipgram", "cbow"),
                    epoch:DiscreteValue(min=5, max=25),
                    lr:ContinuousValue(0.1,1.0)
        ):
        self.min_subword = min_subword
        self.max_subword = max_subword
        self.dimension = dimension
        self.repr_model = model
        self.epoch = epoch
        self.lr = lr
        self.model = None

    def fit(self, X: Seq[Sentence], y=None):
        file = f'uwr_test_{time_ns()}.test'
        with open(file, 'w') as f:
            f.writelines(X)
        
        self.model = fasttext.train_unsupervised(
                                file, self.repr_model,
                                minn=self.min_subword,
                                maxn=self.max_subword, 
                                dim=self.dimension,
                                epoch=self.epoch,
                                lr=self.lr,
        )

        remove(file)
        return self

    def transform(self, _, y):
        return [self.model.get_word_vector(x) for x in y]

    def fit_transform(self, X: Seq[Sentence], y: Seq[Word]):
        self.fit(X, y)
        return self.transform(X, y)
    
    def run(self, corpus:Seq[Sentence], inputs:Seq[Word]) -> MatrixContinuousDense :
        self.fit_transform(X, y)