import numpy as np
from sklearn import svm
from sklearn.naive_bayes import GaussianNB


class StackingEnsemble:

    def __init__(self, X, y):
       self.network = []
       self.X = X
       self.y = y

    def layer_constructor(self,  estimators = ['svm', 'gnb']):
        """
        Creates a layer containing the different estimators in use.
        """
        layer = []
        for estimator in estimators:
            if estimator == 'svm':
               layer.append(svm.SVC()) 
            elif estimator == 'gnb':
                layer.append(GaussianNB())
        return layer
    
    def network_constructor(self, layer): 
        """
        Creates a network containing layers of estimators.
        """
        network = self.network
        network.append(layer)
        return network

    def forward_pass(self):
        """
        Do a forward pass of the stacked network
        """
        X = self.X
        y = self.y
        network = self.network

        output = y
        input_current_layer = []
        input_next_layer = []

        for index, layer in enumerate(network):

            if index == 0:
                input_current_layer = X
                for estimator in layer:
                    estimator.fit(input_current_layer, output)
                    input_next_layer.append(estimator.predict(input_current_layer))
            else :
                input_current_layer = input_next_layer
                input_next_layer = []
                for estimator in layer:
                    estimator.fit(input_current_layer, output)
                    input_next_layer.append(estimator.predict(input_current_layer))
        
        return network
    
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
                    prediction_current_layer = np.concatenate((prediction_current_layer, estimator.predict(input_current_layer)))
                
                prediction_current_layer = np.reshape(prediction_current_layer, (1,2))


            else:
                input_current_layer = prediction_current_layer
                prediction_current_layer = np.array([])
                for estimator in layer:
                    prediction_current_layer = np.concatenate((prediction_current_layer, estimator.predict(input_current_layer)))
                    
        return(prediction_current_layer)

if __name__ == '__main__': 
    X_train = [[0, 0], [1, 1]]
    y_train = [0, 1]
    X_test = [[2.,2.]]
    se = StackingEnsemble(X_train, y_train)
    layer1 = se.layer_constructor()
    layer2 = se.layer_constructor(['svm'])
    network = se.network_constructor(layer1)
    network = se.network_constructor(layer2)
    trained_network = se.forward_pass()
    print(se.predict(X_test))
