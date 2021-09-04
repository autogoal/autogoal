from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.datasets import make_classification
from autogoal.kb import  Seq, Tensor, Discrete, Continuous, Dense, Supervised, AlgorithmBase, algorithm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from autogoal.kb import SemanticType
from autogoal.grammar import Union



class Stacking(AlgorithmBase):
    def __init__(self,level0AlgorithmsA : algorithm(Tensor[2, Continuous, Dense],Supervised[Tensor[1, Discrete, Dense]],Tensor[1, Discrete, Dense]),level0Algorithms : algorithm(Tensor[2, Continuous, Dense],Supervised[Tensor[1, Discrete, Dense]],Tensor[1, Discrete, Dense]),level1Algorithm : algorithm(Tensor[2, Continuous, Dense],Supervised[Tensor[1, Discrete, Dense]],Tensor[1, Discrete, Dense])):
        self.level0 = list()
        self.level0.append(('a',level0Algorithms))
        self.level0.append(('A',level0AlgorithmsA))
        self.level1 = level1Algorithm
        self.model = self.get_stacking(self.level0,self.level1)
        self.mode = "train"

    def get_stacking(self,level0,level1):
        model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
        return model

    def run(self,X : Tensor[2, Continuous, Dense],y : Supervised[Tensor[1, Discrete, Dense]]) -> Tensor[1, Discrete, Dense]:
        if self._mode == "train":
            return self._train(X,y)
        elif self._mode == "eval":
            return self._eval(X,y)

    def train(self):
        self._mode = "train"

    def eval(self):
        self._mode = "eval"

    def _train(self,X,y):
        self.model.fit(X,y)
        return y

    def _eval(self, X, y=None):
        return self.model.predict(X)





