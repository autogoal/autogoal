from autogoal.contrib.keras import KerasSequenceClassifier
from autogoal.contrib.torch import BertEmbedding
from autogoal.datasets import haha
from autogoal.kb import CategoricalVector, List, Sentence, Tuple
from autogoal.ml import AutoClassifier
from autogoal.search import ConsoleLogger, ProgressLogger


classifier = AutoClassifier(
    input=List(Sentence()),
    registry=[KerasSequenceClassifier, BertEmbedding],
    search_kwargs=dict(memory_limit=0, evaluation_timeout=0),
)


Xtrain, Xtest, ytrain, ytest = haha.load()
classifier.fit(Xtrain, ytrain, logger=[ConsoleLogger(), ProgressLogger()])
