import warnings


def find_classes(include=".*", exclude=None):
    result = []

    try:
        from autogoal.contrib.sklearn import find_classes as f
        result.extend(f(include, exclude))
    except ImportError as e:
        warnings.warn(repr(e))
        warnings.warn("Skipping `scikit-learn`. Run `pip install autogoal[sklearn]` to include it.")
        pass

    try:
        from autogoal.contrib.nltk import find_classes as f
        result.extend(f(include, exclude))
    except ImportError as e:
        warnings.warn(repr(e))
        warnings.warn("Skipping `nltk`. Run `pip install autogoal[nltk]` to include it.")
        pass

    return result
