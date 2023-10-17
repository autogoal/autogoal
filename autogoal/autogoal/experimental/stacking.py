# from autogoal.contrib import ensemble
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


class StackingEnsemble:
    def __init__(self, layers=None, final=None):

        if layers == None:
            self.layers = [[SVC(), LogisticRegression()], [DecisionTreeClassifier()]]

        else:
            self.layers = layers

        if final == None:
            self.final = GaussianNB()

        else:
            self.final = final

        self.network = []

    def network_constructor(self):
        """
        Creates a network containing layers of estimators.
        """
        network = self.network
        layers = self.layers
        final = self.final
        network.append(layers)
        network.append(final)
        return network

    def forward_pass(self, X, y):
        """
        Do a forward pass of the stacked network
        """

        network = self.network_constructor()

        output = y
        input_current_layer = []
        input_next_layer = []

        for index, layer in enumerate(network):

            if index == 0:
                input_current_layer = X
                for estimator in layer:
                    estimator.fit(input_current_layer, output)
                    input_next_layer.append(estimator.predict(input_current_layer))
            else:
                input_current_layer = input_next_layer
                input_next_layer = []
                for estimator in layer:
                    estimator.fit(input_current_layer, output)
                    input_next_layer.append(estimator.predict(input_current_layer))

        return network

    def fit(self, X, y):
        input_length = len(X)
        target_lenght = len(y)
        if input_length == target_lenght:
            return self.forward_pass(X, y)
        else:
            raise ValueError("X and y must have the same length")

    def predict(self, X):
        """
        Do a prediction for a test data
        """
        network = self.network
        prediction_current_layer = np.array([])
        input_current_layer = []
        for index, layer in enumerate(network):

            if index == 0:
                input_current_layer = X
                for estimator in layer:
                    prediction_current_layer = np.concatenate(
                        (
                            prediction_current_layer,
                            estimator.predict(input_current_layer),
                        )
                    )

                prediction_current_layer = np.reshape(prediction_current_layer, (1, 2))

            else:
                input_current_layer = prediction_current_layer
                prediction_current_layer = np.array([])
                for estimator in layer:
                    prediction_current_layer = np.concatenate(
                        (
                            prediction_current_layer,
                            estimator.predict(input_current_layer),
                        )
                    )

        return prediction_current_layer


if __name__ == "__main__":
    X_train = [[0, 0], [1, 1]]
    y_train = [0, 1]
    X_test = [[2.0, 2.0]]
    y_test = [1]
    ensemble = StackingEnsemble([SVC(), DecisionTreeClassifier()], [SVC()])
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
