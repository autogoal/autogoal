from autogoal.contrib.sklearn import SklearnClassifier
from autogoal.grammar import  generate_cfg
from autogoal.search import RandomSearch, ProgressLogger

from sklearn.datasets import make_classification


g =  generate_cfg(SklearnClassifier)
X, y = make_classification()

print(g)

def fitness(pipeline):
    pipeline.fit(X, y)
    return pipeline.score(X, y)


search = RandomSearch(g, fitness, random_state=0, errors='warn')
search.run(1000, logger=ProgressLogger())
