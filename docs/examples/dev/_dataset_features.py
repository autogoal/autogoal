import pprint
import json

from autogoal.ml import DatasetFeatureExtractor
from autogoal.datasets import cars, abalone, german_credit, cifar10, haha, meddocan


# for dataset in [cars, abalone, german_credit, cifar10, haha, meddocan]:
#     X, y, *_ = dataset.load()
#     features = DatasetFeatureExtractor().extract_features(X, y)
#     pprint.pprint(dict(dataset=dataset.__name__, features=features))


# Testing that AutoML saves model

from autogoal.ml import AutoML, DatasetFeatureLogger, LearnerMedia, DatasetFeatureExtractor
from autogoal.ml._metalearning import SolutionInfo
from autogoal.contrib.keras import KerasClassifier

extractor = DatasetFeatureExtractor()

X, y = cars.load()
X2, y2 = german_credit.load()

# logger = DatasetFeatureLogger(X, y)

# automl = AutoML()
# automl.fit(X, y, logger=logger)

with open("metalearning.json") as fp:
    solutions = [SolutionInfo.from_dict(json.loads(s)) for s in fp]

learner = LearnerMedia(extractor.extract_features(X, y), solutions)
learner.compute_all_features()
