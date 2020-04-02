import abc
import functools
import warnings


from autogoal.search import Logger


class DatasetFeatureLogger(Logger):
    def __init__(self, X, y=None, extractor=None):
        self.extractor = extractor or DatasetFeatureExtractor()
        self.X = X
        self.y = y

    def begin(self, generations, pop_size):
        self.dataset_features_ = self.extractor.extract_features(self.X, self.y)

    def eval_solution(self, solution, fitness):
        if not hasattr(solution, "sampler_"):
            raise ("Cannot log if the underlying algorithm is not PESearch")

        sampler = solution.sampler_
        print(sampler._updates)
        print(sampler._model)


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
   

