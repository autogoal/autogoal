```python
from autogoal.contrib.nltk import SklearnNLPClassifier
from autogoal.grammar import generate_cfg
from autogoal.search import RandomSearch, EnlightenLogger
from nltk.corpus import movie_reviews
import random
import enlighten

def load_movie_reviews(amount = 2000):
    sentences = []
    classes = []
    ids = list(movie_reviews.fileids())
    random.shuffle(ids)
    size = min([amount, len(ids)])
    
    print("loading movie_review")
    
    counter = 0
    for fd in ids:
        if fd.startswith('neg/'):
            cls = 'neg'
        else:
            cls = 'pos'

        fp = movie_reviews.open(fd)
        sentences.append(fp.read())
        classes.append(cls)
        counter+=1
        if counter >= size:
            break
    
    print('loaded sentences:', len(sentences))
    return sentences, classes

g = generate_cfg(SklearnNLPClassifier)
X, y = load_movie_reviews(100)

def fitness(pipeline):
    pipeline.fit(X, y)
    score = pipeline.score(X, y)
    return score


search = RandomSearch(g, fitness, random_state=0, errors='warn', evaluation_timeout=10)
result = search.run(100, logger=EnlightenLogger())
```

