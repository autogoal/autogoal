def find_classes(include=".*", exclude=None, modules=None):
    import inspect
    import warnings
    import re

    result = []

    if include:
        include = f".*({include}).*"
    else:
        include = r".*"

    if exclude:
        exclude = f".*({exclude}).*"

    if modules is None:
        modules = []

        try:
            from autogoal.contrib import sklearn
            modules.append(sklearn)
        except ImportError as e:
            warnings.warn(repr(e))
            warnings.warn(
                "Skipping `scikit-learn`. Run `pip install autogoal[sklearn]` to include it."
            )
            pass

        try:
            from autogoal.contrib import nltk
            modules.append(nltk)
        except ImportError as e:
            warnings.warn(repr(e))
            warnings.warn(
                "Skipping `nltk`. Run `pip install autogoal[nltk]` to include it."
            )
            pass

        try:
            from autogoal.contrib import gensim
            modules.append(gensim)
        except ImportError as e:
            warnings.warn(repr(e))
            warnings.warn(
                "Skipping `gensim`. Run `pip install autogoal[gensim]` to include it."
            )
            pass

        try:
            from autogoal.contrib import keras
            modules.append(keras)
        except ImportError as e:
            warnings.warn(repr(e))
            warnings.warn(
                "Skipping `keras`. Run `pip install autogoal[keras]` to include it."
            )
            pass

        try:
            from autogoal.contrib import torch
            modules.append(torch)
        except ImportError as e:
            warnings.warn(repr(e))
            warnings.warn(
                "Skipping `torch`. Run `pip install autogoal[torch]` to include it."
            )
            pass

        try:
            from autogoal.contrib import spacy
            modules.append(spacy)
        except ImportError as e:
            warnings.warn(repr(e))
            warnings.warn(
                "Skipping `spacy`. Run `pip install autogoal[spacy]` to include it."
            )
            pass

        try:
            from autogoal.contrib import wikipedia
            modules.append(wikipedia)
        except ImportError as e:
            warnings.warn(repr(e))
            warnings.warn(
                "Skipping `wikipedia`. Run `pip install autogoal[wikipedia]` to include it."
            )
            pass

        from autogoal.contrib import _wrappers
        modules.append(_wrappers)

        from autogoal.contrib import regex
        modules.append(regex)

    for module in modules:
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if not hasattr(cls, 'run'):
                continue

            if not re.match(include, repr(cls)):
                continue

            if exclude is not None and re.match(exclude, repr(cls)):
                continue
            
            result.append(cls)

    return result


__all__ = ['find_classes']
