import pytest
from autogoal.contrib.ensemble._stacking import StackingEnsemble
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def test_stacking(): 
    X_train = [[0, 0], [1, 1]]
    y_train = [0, 1]
    X_test = [[2.,2.]]
    y_test = [1]
    ensemble = StackingEnsemble([SVC(), DecisionTreeClassifier()], [SVC()])
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    assert y_pred == y_test
