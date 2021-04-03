def find_classes(include=None, exclude=None, modules=None, input=None, output=None):
    import inspect
    import re

    result = []

    if include:
        include = f".*({include}).*"
    else:
        include = r".*"

    if exclude:
        exclude = f".*({exclude}).*"

    if input:
        input = f".*({input}).*"

    if output:
        output = f".*({output}).*"

    if modules is None:
        modules = []

        try:
            from autogoal.contrib import sklearn

            modules.append(sklearn)
        except ImportError as e:
            pass

        try:
            from autogoal.contrib import nltk

            modules.append(nltk)
        except ImportError as e:
            pass

        try:
            from autogoal.contrib import gensim

            modules.append(gensim)
        except ImportError as e:
            pass

        try:
            from autogoal.contrib import keras

            modules.append(keras)
        except ImportError as e:
            pass

        try:
            from autogoal.contrib import torch

            modules.append(torch)
        except ImportError as e:
            pass

        try:
            from autogoal.contrib import spacy

            modules.append(spacy)
        except ImportError as e:
            pass

        try:
            from autogoal.contrib import wikipedia

            modules.append(wikipedia)
        except ImportError as e:
            pass

        from autogoal.contrib import wrappers

        modules.append(wrappers)

        from autogoal.contrib import regex

        modules.append(regex)

    for module in modules:
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if not hasattr(cls, "run"):
                continue

            if cls.__name__.startswith("_"):
                continue

            if not re.match(include, repr(cls)):
                continue

            if exclude is not None and re.match(exclude, repr(cls)):
                continue

            if not cls.__module__.startswith("autogoal.contrib"):
                continue

            sig = inspect.signature(cls.run)

            if input and not re.match(input, str(sig.parameters["input"].annotation)):
                continue

            if output and not re.match(output, str(sig.return_annotation)):
                continue

            result.append(cls)

    return result


import enum


class ContribStatus(enum.Enum):
    RequiresDependency = enum.auto()
    RequiresDownload = enum.auto()
    Ready = enum.auto()


def status():
    status = {}
    modules = []

    try:
        from autogoal.contrib import sklearn

        modules.append(sklearn)
    except ImportError as e:
        status["autogoal.contrib.sklearn"] = ContribStatus.RequiresDependency

    try:
        from autogoal.contrib import nltk

        modules.append(nltk)
    except ImportError as e:
        status["autogoal.contrib.nltk"] = ContribStatus.RequiresDependency

    try:
        from autogoal.contrib import gensim

        modules.append(gensim)
    except ImportError as e:
        status["autogoal.contrib.gensim"] = ContribStatus.RequiresDependency

    try:
        from autogoal.contrib import keras

        modules.append(keras)
    except ImportError as e:
        status["autogoal.contrib.keras"] = ContribStatus.RequiresDependency

    try:
        from autogoal.contrib import torch

        modules.append(torch)
    except ImportError as e:
        status["autogoal.contrib.torch"] = ContribStatus.RequiresDependency

    try:
        from autogoal.contrib import spacy

        modules.append(spacy)
    except ImportError as e:
        status["autogoal.contrib.spacy"] = ContribStatus.RequiresDependency

    try:
        from autogoal.contrib import wikipedia

        modules.append(wikipedia)
    except ImportError as e:
        status["autogoal.contrib.wikipedia"] = ContribStatus.RequiresDependency

    for module in modules:
        if hasattr(module, "status"):
            status[module.__name__] = module.status()
        else:
            status[module.__name__] = ContribStatus.Ready

    return status


def download(contrib: str):
    modules = {}

    try:
        from autogoal.contrib import sklearn

        modules["sklearn"] = sklearn
    except ImportError as e:
        pass

    try:
        from autogoal.contrib import nltk

        modules["nltk"] = nltk
    except ImportError as e:
        pass

    try:
        from autogoal.contrib import gensim

        modules["gensim"] = gensim
    except ImportError as e:
        pass

    try:
        from autogoal.contrib import keras

        modules["keras"] = keras
    except ImportError as e:
        pass

    try:
        from autogoal.contrib import torch

        modules["torch"] = torch
    except ImportError as e:
        pass

    try:
        from autogoal.contrib import spacy

        modules["spacy"] = spacy
    except ImportError as e:
        pass

    try:
        from autogoal.contrib import wikipedia

        modules["wikipedia"] = wikipedia
    except ImportError as e:
        pass

    if contrib not in modules:
        raise ValueError(f"Contrib `{contrib}` cannot be imported.")

    contrib = modules[contrib]

    if not hasattr(contrib, "download"):
        return False

    return contrib.download()


__all__ = ["find_classes", "status", "download"]
