import uuid
import abc
import functools
import warnings
import json
import collections
import numpy as np

from typing import List
from autogoal.search import Logger
from autogoal.utils import nice_repr
from autogoal import sampling
# from sklearn.feature_extraction import DictVectorizer


class DatasetFeatureLogger(Logger):
    def __init__(
        self,
        X,
        y=None,
        extractor=None,
        output_file="metalearning.json",
        problem_features=None,
        environment_features=None,
    ):
        self.extractor = extractor or DatasetFeatureExtractor()
        self.X = X
        self.y = y
        self.run_id = str(uuid.uuid4())
        self.output_file = output_file
        self.problem_features = problem_features or {}
        self.environment_features = environment_features or {}

    def begin(self, generations, pop_size):
        self.dataset_features_ = self.extractor.extract_features(self.X, self.y)

    def eval_solution(self, solution, fitness):
        if not hasattr(solution, "sampler_"):
            raise ("Cannot log if the underlying algorithm is not PESearch")

        sampler = solution.sampler_

        features = {k: v for k, v in sampler._updates.items() if isinstance(k, str)}
        feature_types = {k: repr(v) for k, v in sampler._model.items() if k in features}

        info = SolutionInfo(
            uuid=self.run_id,
            fitness=fitness,
            problem_features=dict(self.dataset_features_, **self.problem_features),
            environment_features=dict(self.environment_features),
            pipeline_features=features,
            feature_types=feature_types,
        ).to_dict()

        with open(self.output_file, "a") as fp:
            fp.write(json.dumps(info) + "\n")


class DatasetFeatureExtractor:
    def __init__(self, features_extractors=None):
        self.feature_extractors = list(features_extractors or _EXTRACTORS)

    def extract_features(self, X, y=None):
        features = {}

        for extractor in self.feature_extractors:
            features.update(**extractor(X, y))

        return features


_EXTRACTORS = []


def feature_extractor(func):
    @functools.wraps(func)
    def wrapper(X, y=None):
        try:
            result = func(X, y)
        except:
            result = None
            # raise

        return {func.__name__: result}

    _EXTRACTORS.append(wrapper)
    return wrapper


# Feature extractor methods


@feature_extractor
def is_supervised(X, y=None):
    return y is not None


@feature_extractor
def dimensionality(X, y=None):
    d = 1

    for di in X.shape[1:]:

        d *= di

    return d


@feature_extractor
def training_examples(X, y=None):
    try:
        return X.shape[0]
    except:
        return len(X)


@feature_extractor
def has_numeric_features(X, y=None):
    return any([xi for xi in X[0] if isinstance(xi, (float, int))])


@feature_extractor
def numeric_variance(X, y=None):
    return X.std()


@feature_extractor
def average_number_of_words(X, y=None):
    return sum(len(sentence.split(" ")) for sentence in X) / len(X)


@feature_extractor
def has_text_features(X, y=None):
    return isinstance(X[0], str)


@nice_repr
class SolutionInfo:
    def __init__(
        self,
        uuid: str,
        problem_features: dict,
        pipeline_features: dict,
        environment_features: dict,
        feature_types: dict,
        fitness: float,
    ):
        self.problem_features = problem_features
        self.pipeline_features = pipeline_features
        self.environment_features = environment_features
        self.feature_types = feature_types
        self.fitness = fitness
        self.uuid = uuid

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(d):
        return SolutionInfo(**d)


class LearnerMedia:
    def __init__(self, problem, solutions: List[SolutionInfo], beta=1):
        self.solutions = solutions
        self.problem = problem
        self.beta = beta

    def initialize(self):
        raise NotImplementedError("We need to refactor to not depend on DictVectorizer")

        self.best_fitness = collections.defaultdict(lambda: 0)
        self.all_features = {}

        for i in self.solutions:
            self.best_fitness[i.uuid] = max(self.best_fitness[i.uuid], i.fitness)

            for feature in i.pipeline_features:
                self.all_features[feature] = None

        self.vect = DictVectorizer(sparse=False)
        self.vect.fit([self.problem])

        self.weights_solution = self.calculate_weight_examples(self.solutions)

    def compute_all_features(self):
        self.initialize()

        for feature in list(self.all_features):
            self.all_features[feature] = self.compute_feature(feature)
            print(feature, "=", self.all_features[feature])

    def compute_feature(self, feature):
        """Select for training all solutions where is used the especific feature.
    
        Predict the media of the parameter value.
        """
        # find the relevant solutions, that contain the production to predict
        important_solutions = []
        feature_prototype = None

        for i, w in zip(self.solutions, self.weights_solution):
            if feature in i.pipeline_features:
                for value in i.pipeline_features[feature]:
                    important_solutions.append((value, w))

                if feature_prototype is None:
                    feature_prototype = eval(
                        i.feature_types[feature], sampling.__dict__, {}
                    )

        if feature_prototype is None:
            return None

        return feature_prototype.weighted(important_solutions)

    def calculate_weight_examples(self, solutions: List[SolutionInfo]):
        """Calcule a weight of each example considering the fitness and the similariti with the
        actual problem 
        """
        # met = fitness * (similarity)^beta
        # métrica utilizada en active learning para combinar informativeness with representativeness

        weights = []

        for info in solutions:
            # normalize fitness
            info.fitness = self.normalize_fitness(info)

            if info.fitness == 0:
                continue

            # calculate similarity
            sim = self.similarity_cosine(info.problem_features)
            # calculate metric for weight
            weights.append(info.fitness * (sim) ** self.beta)

        return weights

    def normalize_fitness(self, info: SolutionInfo):
        """Normalize the fitness with respect to the best solution in the problem where that solution is evaluated
        """
        return info.fitness / self.best_fitness[info.uuid]

    def similarity_cosine(self, other_problem):
        """Caculate the cosine similarity for a particular solution problem(other problem) 
        and the problem analizing
        """
        x = self.vect.transform(other_problem)[0]
        y = self.vect.transform(self.problem)[0]

        return np.dot(x, y) / (np.dot(x, x) ** 0.5 * np.dot(y, y) ** 0.5)

    def similarity_learning(self, other_problem):
        """ Implementar una espicie de encoding para los feature de los problemas
        """
        raise NotImplementedError()
