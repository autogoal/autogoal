import warnings


def find_classes(include=".*", exclude=None):
    result = []

    try:
        from autogoal.contrib.sklearn import find_classes as f
        result.extend(f(include, exclude))
    except ImportError:
        warnings.warn("Skipping `scikit-learn`. Run `pip install autogoal[sklearn]` to include it.")
        pass

    return result
