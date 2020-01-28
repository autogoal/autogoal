from autogoal.contrib.keras import KerasSequenceClassifier
from autogoal.contrib.torch import BertEmbedding

from autogoal.datasets import movie_reviews
from autogoal.search import RandomSearch

bert = BertEmbedding(10)
x = bert.run("this is an english sentence for bert")

print(x)

classifier = KerasSequenceClassifier(100, 768)
classifier.sample()

classifier.model.summary()

classifier.predict([x])