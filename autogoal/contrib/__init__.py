import warnings
import re


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

    try:
        from autogoal.contrib.spacy import find_classes as f

        result.extend(f(include, exclude))
    except ImportError as e:
        warnings.warn(repr(e))
        warnings.warn(
            "Skipping `spacy`. Run `pip install autogoal[spacy]` to include it."
        )
        pass

    try:
        from autogoal.contrib.wikipedia import find_classes as f

        result.extend(f(include, exclude))
    except ImportError as e:
        warnings.warn(repr(e))
        warnings.warn(
            "Skipping `wikipedia`. Run `pip install autogoal[wikipedia]` to include it."
        )
        pass

    from autogoal.contrib._wrappers import (
        MatrixBuilder,
        VectorAggregator,
        TensorBuilder,
        FlagsMerger,
        SentenceFeatureExtractor,
        DocumentFeatureExtractor,
        MultipleFeatureExtractor,
    )

    result.extend(
        [
            MatrixBuilder,
            VectorAggregator,
            TensorBuilder,
            FlagsMerger,
            SentenceFeatureExtractor,
            DocumentFeatureExtractor,
            MultipleFeatureExtractor,
        ]
    )

    return [
        cls
        for cls in result
        if re.match(include, repr(cls))
        and (exclude is None or not re.match(exclude, repr(cls)))
    ]
