import pprint

from autogoal.ml import DatasetFeatureExtractor
from autogoal.datasets import cars, abalone, german_credit, cifar10, haha, meddocan


# for dataset in [cars, abalone, german_credit, cifar10, haha, meddocan]:
#     X, y, *_ = dataset.load()
#     features = DatasetFeatureExtractor().extract_features(X, y)
#     pprint.pprint(dict(dataset=dataset.__name__, features=features))


# Testing that AutoML saves model

from autogoal.ml import AutoML, DatasetFeatureLogger


X, y = cars.load()

logger = DatasetFeatureLogger(X, y)

automl = AutoML()
automl.fit(X, y, logger=logger)
