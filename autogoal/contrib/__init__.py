import warnings


def find_classes(include=".*", exclude=None):
    result = []

    try:
        from autogoal.contrib.sklearn import find_classes as f

        result.extend(f(include, exclude))
    except ImportError as e:
        warnings.warn(repr(e))
        warnings.warn(
            "Skipping `scikit-learn`. Run `pip install autogoal[sklearn]` to include it."
        )
        pass

    try:
        from autogoal.contrib.nltk import find_classes as f

        result.extend(f(include, exclude))
    except ImportError as e:
        warnings.warn(repr(e))
        warnings.warn(
            "Skipping `nltk`. Run `pip install autogoal[nltk]` to include it."
        )
        pass

    try:
        from autogoal.contrib.gensim import find_classes as f

        result.extend(f(include, exclude))
    except ImportError as e:
        warnings.warn(repr(e))
        warnings.warn(
            "Skipping `gensim`. Run `pip install autogoal[gensim]` to include it."
        )
        pass

    try:
        from autogoal.contrib.keras import find_classes as f

        result.extend(f(include, exclude))
    except ImportError as e:
        warnings.warn(repr(e))
        warnings.warn(
            "Skipping `keras`. Run `pip install autogoal[keras]` to include it."
        )
        pass

    try:
        from autogoal.contrib.torch import find_classes as f

        result.extend(f(include, exclude))
    except ImportError as e:
        warnings.warn(repr(e))
        warnings.warn(
            "Skipping `torch`. Run `pip install autogoal[torch]` to include it."
        )
        pass

    from autogoal.contrib._wrappers import (
        MatrixBuilder,
        VectorAggregator,
        TensorBuilder,
    )

    result.extend([MatrixBuilder, VectorAggregator, TensorBuilder])

    return result
