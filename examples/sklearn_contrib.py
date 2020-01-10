from autogoal.contrib.sklearn import SklearnClassifier
from autogoal.grammar import generate_cfg


g = generate_cfg(SklearnClassifier)
print(g)

print(g.sample())
