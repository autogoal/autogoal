from autogoal.contrib.nltk._generated import AffixTagger
from autogoal.datasets.meddocan import load


X, _, y, _ = load()

# for sent in X:
#     print(repr(sent))

X = X[:10]
y = y[:10]

tagger = AffixTagger(-3, 2, 0)
tagger.run((X, y))
tagger.eval()
ypred = tagger.run((X, None))

print("ypred", ypred)
print("ytrue", y)
