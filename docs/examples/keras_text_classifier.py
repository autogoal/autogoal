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

# bert = BertEmbedding(32)
# x = bert.run(
#     [
#         "this is an english sentence for bert",
#         "another sentence",
#         "a longer sentence that some words will be removed for sure, not really, but now it is",
#     ]
# )

# print(x.shape)

# classifier = KerasSequenceClassifier(None, 768)

# # for i in range(10):
# classifier.sample()
# classifier.model.summary()

# classifier.fit(x, np.asarray([True, False, True]))
