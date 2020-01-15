try:
    import sklearn

    major, minor, *rest = sklearn.__version__.split(".")
    assert int(major) == 0 and int(minor) >= 22
except:
    print("(!) Code in `autogoal.contrib.sklearn` requires `sklearn=^0.22`.")
    print("(!) You can install it with `pip install autogoal[sklearn]`.")
    raise


import inspect
import re


def find_classes(include=".*", exclude=None):
    """
    Returns the list of all `scikit-learn` wrappers in `autogoal`.

    You can pass filters to include or exclude specific classes.

    ##### Parameters

    - `include`: list of


    ##### Examples

    ```python
    >>> find_classes(include='.*Classifier', exclude='DecisionTree.*')

    ```
    """
    import autogoal.contrib.sklearn._generated as module
    from autogoal.contrib.sklearn._builder import SklearnEstimator, SklearnTransformer

    return [
        c
        for n, c in inspect.getmembers(
            module,
            lambda c: inspect.isclass(c)
            and issubclass(c, (SklearnEstimator, SklearnTransformer))
            and re.match(include, c.__name__)
            and (exclude is None or not re.match(exclude, c.__name__)),
        )
    ]
