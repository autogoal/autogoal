from autogoal.ml import AutoTextClassifier
from autogoal.datasets import haha

classifier = AutoTextClassifier()
X_train, X_test, y_train, y_test = haha.load()

classifier.fit(X_train, y_train)
classifier.score(X_test, y_test)
