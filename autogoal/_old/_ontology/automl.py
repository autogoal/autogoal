# coding: utf8

import importlib
import random
import warnings
from pprint import pprint

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split as split

from ..optimization import PGE, Grammar, Individual
from ._nltk import NLTKResolver
from ._sklearn import SklearnResolver
from .data import solve_type
from .grammar import get_grammar
from .ontology import onto


class Pipeline:
    def __init__(self, algorithms):
        self.algorithms = algorithms

    def train(self, X, y=None):
        for alg in self.algorithms:
            X = alg.train(X, y)

        return X

    def run(self, X):
        for alg in self.algorithms:
            X = alg.run(X)

        return X

    def __repr__(self):
        return "Pipeline({})".format(repr(self.algorithms))


INSTANCE_RESOLVERS = {onto.ScikitLearn: SklearnResolver(), onto.NLTK: NLTKResolver()}


class OntoGrammar(Grammar):
    def __init__(
        self,
        X,
        y,
        input_type,
        output_type,
        score_function,
        split_factor=0.7,
        depth=2,
        include=[],
        exclude=[],
    ):
        self.X = X
        self.y = y
        self.input_type = input_type
        self.output_type = output_type
        self.split_factor = split_factor
        self.score_function = score_function
        self.include = include
        self.exclude = exclude
        self.depth = depth
        super().__init__()

    def grammar(self):
        return get_grammar(
            self.input_type,
            self.output_type,
            depth=self.depth,
            include=self.include,
            exclude=self.exclude,
        )

    def evaluate(self, pipeline, cmplx=1.0):
        X, y = self.X, self.y

        if cmplx < 1.0:
            X, _, y, _ = split(X, y, train_size=cmplx)

        Xtrain, Xtest, ytrain, ytest = split(X, y, train_size=self.split_factor)

        pipeline.train(Xtrain, ytrain)
        ypred = pipeline.run(Xtest)

        return self.score_function(ytest, ypred)

    def generate(self, ind: Individual):
        pipeline = ind.sample()
        pipeline = list(self._flatten(pipeline))

        return Pipeline([self._resolve(ins, params) for ins, params in pipeline])

    def _resolve(self, instance, parameters):
        return INSTANCE_RESOLVERS[instance.implementedIn].resolve(instance, parameters)

    def _flatten(self, pipeline):
        root = list(pipeline.keys())[0]
        onto_obj = getattr(onto, root)

        if onto_obj is None or onto_obj.implementedIn is None:
            children = pipeline[root]

            for c in children:
                yield from self._flatten(c)
        else:
            clss = onto_obj
            parameters = [list(d.items())[0] for d in pipeline[root]]
            parameters = {
                key.split("__")[1]: self._flatten_value(value)
                for key, value in parameters
            }
            yield (clss, parameters)

    def _flatten_value(self, value):
        if isinstance(value, list):
            assert len(value) == 1
            value = value[0]

        if value in ["yes", "no"]:
            return value == "yes"

        return value


class AutoML:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fitted = False
        self.verbose = kwargs.get("verbose", False)

    def optimize(self, X, y, input=None, output=None):
        if self.fitted:
            warnings.warn("Already fitted, will override previous values.")

        self.fitted = True
        self.input_type_ = input or solve_type(X)
        self.output_type_ = output or solve_type(y)

        if self.verbose:
            print("Input=%s, Output=%s" % (self.input_type_, self.output_type_))

        kwargs = self.kwargs

        self.grammar = OntoGrammar(
            X,
            y,
            self.input_type_,
            self.output_type_,
            accuracy_score,
            depth=kwargs.pop("depth", 2),
            include=kwargs.pop("include", []),
            exclude=kwargs.pop("exclude", []),
        )
        pge = PGE(self.grammar, **kwargs)
        self.best_ = pge.run(100)

    def run(self, X):
        return self.best_.run(X)

    def score(self, X, y):
        return accuracy_score(self.run(X), y)


def main(args):
    from .hmlopt.datasets.uci import car, german_credit, wine_quality
    from .hmlopt.examples import movie_reviews

    from sklearn.model_selection import train_test_split

    datasets = {
        "car": car.load_corpus,
        "german": german_credit.load_corpus,
        "wine": lambda: wine_quality.load_corpus(white=True),
        "movies": lambda: movie_reviews.load_corpus(easy=True),
    }

    X, y = datasets[args.dataset]()

    include = [onto[i] for i in args.include]
    exclude = [onto[i] for i in args.exclude]

    input = onto[args.input]
    output = onto[args.output]

    random.seed(args.seed)

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.7)

    automl = AutoML(
        errors=args.errors,
        verbose=args.verbose,
        include=include,
        exclude=exclude,
        timeout=args.timeout,
    )
    automl.optimize(Xtrain, ytrain, input=input, output=output)

    print(automl.score(Xtest, ytest))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="car")
    parser.add_argument("--include", nargs="+", default=[])
    parser.add_argument("--exclude", nargs="+", default=[])
    parser.add_argument("--errors", default="warn")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=300)

    args = parser.parse_args()
    main(args)
