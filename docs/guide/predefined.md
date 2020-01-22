```python
from autogoal.ml import AutoTextClassifier
from autogoal.datasets import movie_reviews

classifier = AutoTextClassifier(include_filter=r"(.*Classifier|.*Vectorizer)")
X, y = movie_reviews.load(max_examples=100)

classifier.fit(X, y)

print(classifier.best_pipeline_)
print(classifier.best_score_)
```

