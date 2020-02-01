from autogoal.contrib.keras import KerasSequenceClassifier
from autogoal.contrib.torch import BertEmbedding
from autogoal.datasets import haha
from autogoal.kb import CategoricalVector, List, Sentence, Tuple
from autogoal.ml import AutoML
from autogoal.search import ConsoleLogger, ProgressLogger


classifier = AutoML(
    input=List(Sentence()),
    registry=[KerasSequenceClassifier, BertEmbedding],
    # search_kwargs=dict(memory_limit=4 * 1024 ** 3, evaluation_timeout=60),
    search_kwargs=dict(memory_limit=0, evaluation_timeout=0),
)


Xtrain, Xtest, ytrain, ytest = haha.load(max_examples=10)

# embedding = BertEmbedding()
# tokens = embedding.run(Xtrain)

# classifier = KerasSequenceClassifier().sample()
# classifier.run((tokens, ytrain))

classifier.fit(Xtrain, ytrain, logger=[ConsoleLogger(), ProgressLogger()])
