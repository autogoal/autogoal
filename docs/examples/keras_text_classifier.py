from autogoal.contrib.keras import KerasSequenceClassifier
from autogoal.contrib.torch import BertEmbedding

from autogoal.datasets import movie_reviews
from autogoal.search import RandomSearch

import numpy as np

bert = BertEmbedding(32)
x = bert.run(
    [
        "this is an english sentence for bert",
        "another sentence",
        "a longer sentence that some words will be removed for sure, not really, but now it is",
    ]
)

print(x.shape)

classifier = KerasSequenceClassifier(None, 768)

# for i in range(10):
classifier.sample()
classifier.model.summary()

classifier.fit(x, np.asarray([True, False, True]))
