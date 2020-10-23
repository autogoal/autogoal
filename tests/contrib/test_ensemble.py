import pytest

from autogoal.contrib.ensemble._stacking import StackingEnsemble

def test_stacking(): 
    X_train = [[0, 0], [1, 1]]
    y_train = [0, 1]
    X_test = [[2.,2.]]
    ensemble = StackingEnsemble([SVC(),GaussianNB()], [SVC()])
    print(ensemble.forward_pass(X_train, y_train))
    print(ensemble.predict(X_test))


