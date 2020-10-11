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
            if not hasattr(cls, 'run'):
                continue

            if cls.__name__.startswith("_"):
                continue

            if not re.match(include, repr(cls)):
                continue

            if exclude is not None and re.match(exclude, repr(cls)):
                continue

            sig = inspect.signature(cls.run)

            if input and not re.match(input, str(sig.parameters["input"].annotation)):
                continue

            if output and not re.match(output, str(sig.return_annotation)):
                continue
            
            result.append(cls)

    return result


__all__ = ['find_classes']
