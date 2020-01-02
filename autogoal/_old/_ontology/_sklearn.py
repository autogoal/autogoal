#

import collections
import importlib
import inspect
import re
import types
import warnings

import enlighten
import numpy as np
import sklearn
import sklearn.cluster
import sklearn.cross_decomposition
import sklearn.feature_extraction

from .data import (
    get_data_for,
    is_algorithm,
    is_classifier,
    is_clusterer,
    is_regressor,
    is_transformer,
)


class SklearnResolver:
    def resolve(self, instance, parameters):
        import_code = instance.importCode.split(".")
        module = ".".join(import_code[:-1])
        classname = import_code[-1]

        module = importlib.import_module(module, __package__)
        clss = getattr(module, classname)
        return SklearnAlgorithm(clss(**parameters))


class SklearnAlgorithm:
    def __init__(self, algorithm):
        self.algorithm = algorithm

    def train(self, X, y=None):
        if hasattr(self.algorithm, "fit_transform"):
            return self.algorithm.fit_transform(X, y)
        else:
            self.algorithm.fit(X, y)
            return None

    def run(self, X):
        if hasattr(self.algorithm, "transform"):
            return self.algorithm.transform(X)
        else:
            return self.algorithm.predict(X)

    def __repr__(self):
        return repr(self.algorithm)


class SklearnCountVectorizerWrapper(sklearn.feature_extraction.text.CountVectorizer):
    def __init__(self, **kwargs):
        kwargs["analyzer"] = self.analyzer
        super().__init__(self, **kwargs)

    def analyzer(self, x):
        return x.tokens


def build_ontology_sklearn(onto):
    imports = _walk(sklearn)

    Software = onto["Software"]
    skip_parts = set(["Classes", "Data", "Base", "Cluster"])
    ScikitLearn = Software("ScikitLearn")

    classes = {
        "Classifier": onto["Classifier"],
        "Regressor": onto["Regressor"],
        "Clusterer": onto["Clusterer"],
        "Transformer": onto["Transformer"],
    }

    with onto:
        manager = enlighten.get_manager()
        counter = manager.counter(total=len(imports), unit="classes")

        for cls in imports:
            # print(cls)
            counter.update()

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                base_class = None

                if is_classifier(cls):
                    base_class = "Classifier"
                    estimated_input, estimated_output = is_classifier(cls)

                elif is_regressor(cls):
                    base_class = "Regressor"
                    estimated_input, estimated_output = is_regressor(cls)

                elif is_clusterer(cls):
                    base_class = "Clusterer"
                    estimated_input, estimated_output = is_clusterer(cls)

                elif is_transformer(cls):
                    base_class = "Transformer"
                    estimated_input, estimated_output = is_transformer(cls)

                if base_class:
                    print(cls.__name__, "--> is -->", base_class)
                else:
                    print("!!! Cannot find what is", cls)
                    continue

            parts = _extract_parts(cls)
            parts = [p for p in parts if p not in skip_parts]

            for i, part in enumerate(parts[:-1]):
                # print(part)
                the_class = _make_class(
                    part + base_class,
                    (parts[i - 1] + base_class) if i > 0 else base_class,
                    classes,
                )

            clss = the_class("Sklearn" + cls.__name__)
            clss.implementedIn = ScikitLearn
            clss.importCode = "{}.{}".format(cls.__module__, cls.__qualname__)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                parameters = _get_args(cls)

                for p, meta in sorted(parameters.items(), key=lambda t: t[0]):
                    clss.hasParameter.append(
                        _make_parameter("Sklearn" + cls.__name__, p, meta, onto)
                    )

            clss.hasInput = estimated_input
            clss.hasOutput = estimated_output

        counter.close()
        manager.stop()

    # Create our CV wrapper instance
    CountVectorizer = onto.SklearnCountVectorizer
    CountVectorizerWrapper = CountVectorizer.is_a[0]("SklearnCountVectorizerWrapper")
    CountVectorizerWrapper.implementedIn = ScikitLearn
    CountVectorizerWrapper.importCode = "._sklearn.SklearnCountVectorizerWrapper"
    CountVectorizerWrapper.hasInput = get_data_for(onto.SentenceCorpus, onto.Tokenized)
    CountVectorizerWrapper.hasOutput = get_data_for(
        onto.Matrix, onto.Continuous, onto.Sparse
    )


def _walk(module, name="sklearn"):
    imports = []

    def _walk_p(module, name="sklearn"):
        all_elements = module.__all__

        for elem in all_elements:

            if elem == "exceptions":
                continue

            name = name + "." + elem

            try:
                obj = getattr(module, elem)

                if isinstance(obj, type):
                    if name.endswith("CV"):
                        continue

                    if not is_algorithm(obj):
                        continue

                    imports.append(obj)

                _walk_p(obj, name)
            except:
                pass

            try:
                inner_module = importlib.import_module(name)
                _walk_p(inner_module, name)
            except:
                pass

    _walk_p(module, name)

    imports.sort(key=lambda c: c.__name__)
    return imports


def _extract_parts(cls):
    parts = str(cls).strip("<>'").split(".")[1:]
    parts = [p.title() if p.islower() else p for p in parts]
    parts = ["".join(p.split("_")) for p in parts]
    parts = list(collections.OrderedDict.fromkeys(parts))
    return parts


def _find_parameter_values(parameter, cls):
    documentation = []
    lines = cls.__doc__.split("\n")

    while lines:
        l = lines.pop(0)
        if l.strip().startswith(parameter):
            documentation.append(l)
            tabs = l.index(parameter)
            break

    while lines:
        l = lines.pop(0)

        if not l.strip():
            continue

        if l.startswith(" " * (tabs + 1)):
            documentation.append(l)
        else:
            break

    options = set(re.findall(r"'(\w+)'", " ".join(documentation)))
    valid = []
    invalid = []
    skip = set(["deprecated", "auto_deprecated", "precomputed"])

    for opt in options:
        opt = opt.lower()
        if opt in skip:
            continue
        try:
            cls(**{parameter: opt}).fit(np.ones((10, 10)), [True] * 5 + [False] * 5)
            valid.append(opt)
        except:
            invalid.append(opt)

    return sorted(valid)


def _get_args(cls):
    specs = inspect.getfullargspec(cls.__init__)

    args = specs.args
    specs = specs.defaults

    if not args or not specs:
        return {}

    args = args[-len(specs) :]

    args_map = {k: v for k, v in zip(args, specs)}

    drop_args = [
        "verbose",
        "random_state",
        "n_jobs",
        "max_iter",
        "class_weight",
        "warm_start",
        "copy_X",
        "copy_x",
        "copy",
        "eps",
    ]

    for arg in drop_args:
        args_map.pop(arg, None)

    result = {}

    for arg, value in args_map.items():
        types, values = _get_arg_values(arg, value, cls)
        if not values:
            continue
        result[arg] = dict(type=types.__name__, default=types(value), values=values)

    return result


def _get_arg_values(arg, value, cls):
    if isinstance(value, bool):
        return bool, [False, True]
    if isinstance(value, int):
        return int, _get_integer_values(arg, value, cls)
    if isinstance(value, float):
        return float, _get_float_values(arg, value, cls)
    if isinstance(value, str):
        return str, _find_parameter_values(arg, cls)

    return None, None


def _get_integer_values(arg, value, cls):
    if value == 0:
        min_val = -100
        max_val = 100
    else:
        min_val = value // 2
        max_val = 2 * value

    return min_val, max_val


def _get_float_values(arg, value, cls):
    if value == 0:
        min_val = -1
        max_val = 1
    elif 0 < value <= 0.1:
        min_val = value / 100
        max_val = 1
    elif 0 < value <= 1:
        min_val = 1e-6
        max_val = 1
    else:
        min_val = value / 2
        max_val = 2 * value

    return min_val, max_val


def _make_parameter(name, parameter, meta, onto):
    if meta["type"] == "float":
        param_obj = onto["ContinuousHyperParameter"](name + "__" + parameter)
        param_obj.hasDefaultFloatValue = meta["default"]
        param_obj.hasMinFloatValue = meta["values"][0]
        param_obj.hasMaxFloatValue = meta["values"][1]
        return param_obj

    if meta["type"] == "int":
        param_obj = onto["DiscreteHyperParameter"](name + "__" + parameter)
        param_obj.hasDefaultIntValue = meta["default"]
        param_obj.hasMinIntValue = meta["values"][0]
        param_obj.hasMaxIntValue = meta["values"][1]
        return param_obj

    if meta["type"] == "bool":
        param_obj = onto["BooleanHyperParameter"](name + "__" + parameter)
        param_obj.hasDefaultBoolValue = meta["default"]
        return param_obj

    if meta["type"] == "str":
        param_obj = onto["StringHyperParameter"](name + "__" + parameter)
        param_obj.hasDefaultStringValue = meta["default"]
        for v in meta["values"]:
            param_obj.hasStringValues.append(v)
        return param_obj

    raise ValueError("invalid parameter type", meta["type"])


def _make_class(part, parent, classes):
    if part in classes:
        return classes[part]

    clss = types.new_class(part, (classes[parent],))
    classes[part] = clss
    return clss
