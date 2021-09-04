from autogoal.datasets import cars
from autogoal.kb import (MatrixContinuousDense, 
                         Supervised, 
                         VectorCategorical)
from autogoal.ml import AutoML
from autogoal.contrib import find_classes
from sklearn.datasets import make_classification
from stackingEnsemble import Stacking
from autogoal.kb import  Seq, Tensor, Discrete, Continuous, Dense
from autogoal.utils import Min
from autogoal.search import RichLogger
from autogoal.contrib import sklearn

# Load dataset
X_train, y_train = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)
X_test, y_test = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=1)

# Instantiate AutoML and define input/output types
automl = AutoML(
    input=(Tensor[2, Continuous, Dense],Supervised[Tensor[1, Discrete, Dense]]),
    output=Tensor[1, Discrete, Dense],
    registry=[Stacking] + find_classes(),
    evaluation_timeout=Min
    
)
data = [[2.47475454,0.40165523,1.68081787,2.88940715,0.91704519,-3.07950644,4.39961206,0.72464273,-4.86563631,-6.06338084,-1.22209949,-0.4699618,1.01222748,-0.6899355,-0.53000581,6.86966784,-3.27211075,-6.59044146,-2.21290585,-3.139579]]

# Run the pipeline search process
automl.fit(X_train, y_train)#,logger=[RichLogger()])

score = automl.score(X_test,y_test)
print(f"Score: {score}")

#answer = automl.predict(data)
#print(f"Answers: {answers}")


