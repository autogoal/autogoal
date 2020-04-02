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
from sklearn.feature_extraction import DictVectorizer


class DatasetFeatureLogger(Logger):
    def __init__(self, X, y=None, extractor=None, output_file="metalearning.json"):
        self.extractor = extractor or DatasetFeatureExtractor()
        self.X = X
        self.y = y
        self.run_id = str(uuid.uuid4())
        self.output_file = output_file

    def begin(self, generations, pop_size):
        self.dataset_features_ = self.extractor.extract_features(self.X, self.y)

    def eval_solution(self, solution, fitness):
        if not hasattr(solution, "sampler_"):
            raise ("Cannot log if the underlying algorithm is not PESearch")

        sampler = solution.sampler_

        features = {k: v for k, v in sampler._updates.items() if isinstance(k, str)}

        info = SolutionInfo(
            uuid=self.run_id,
            fitness=fitness,
            problem_features=self.dataset_features_,
            pipeline_features=features,
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


@nice_repr
class SolutionInfo:
    def __init__(
        self,
        uuid: str,
        problem_features: dict,
        pipeline_features: dict,
        fitness: float,
    ):
        self.problem_features = problem_features
        self.pipeline_features = pipeline_features
        self.fitness = fitness
        self.uuid = uuid

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(d):
        return SolutionInfo(**d)


class LearnerMedia:
    def __init__(self, grammar, problem, solutions: List[SolutionInfo], beta=1):
        self.solutions = solutions
        self.problem = problem
        self.beta = beta
        self.grammar = grammar
        self.best_fitness = collections.defaultdict(lambda: 0)

        for i in self.solutions:
            self.best_fitness[i.uuid] = max(self.best_fitness[i.uuid], i.fitness)

        self.vect = DictVectorizer(sparse=False)
        self.vect.fit([problem])

        self.weights_solution = self.calculate_weight_examples(self.solutions)

    def compute_mean(self, feature):
        """Select for training all solutions where is used the especific feature.
    
        Predict the media of the parameter value.
        """
        # find the relevant solutions, that contain the production to predict
        important_solutions = []

        for i, w in zip(self.solutions, self.weights_solution):
            if feature in i.pipeline_features:
                important_solutions.append((i.pipeline_features[feature], w))

        result = 0
        result_w = 0
        for s, w in important_solutions:
            result += s * w
            result_w += w

        return result / result_w

    def calculate_weight_examples(self, solutions: List[SolutionInfo]):
        """Calcule a weight of each example considering the fitness and the similariti with the
        actual problem 
        """
        # met = fitness * (similarity)^beta
        # métrica utilizada en active learning para combinar informativeness with representativeness

        weights = []

        for i in solutions:
            # normalize fitness
            i.fitness = self.normalize_fitness(i)
            # calculate similarity
            simil = self.similarity_cosine(i.problem_features)
            # calculate metric for weight
            weights.append(i.fitness * (simil) ** self.beta)

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

