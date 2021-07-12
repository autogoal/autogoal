from autogoal.kb import Sentence, Seq, Supervised, Categorical
from autogoal.search import RichLogger
from autogoal.utils import Min, Gb
from autogoal.contrib import find_classes
from autogoal.ml import AutoML
from autogoal.datasets import movie_reviews

from autogoal.experimental.gramformer import GramCorrect

X, y = movie_reviews.load() 

automl = AutoML(
    input=(Seq[Sentence], Supervised[Categorical]),
    output=Seq[Sentence],
    registry=[GramCorrect] + find_classes(),
    evaluation_timeout= 1.5 * Min,
    memory_limit= 3 * Gb,
    search_timeout= 2 * Min,
)

automl.fit(X, y, logger=[RichLogger()])
X_correct = automl.predict(X)

#compare results
print("Original: ", X[0])
print('-'*100)
print("Correct: ", X_correct[0])