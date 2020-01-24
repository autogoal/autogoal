import inspect
import re


def find_classes(include=".*", exclude=None):
    """
    Returns the list of all `nltk` wrappers in `autogoal`.

    You can pass filters to include or exclude specific classes.
    The filters are regular expressions that are matched against
    the names of the classes. Only classes that pass the `include` filter
    and not the `exclude` filter will be returned.
    By default all classes are returned.

    ##### Parameters

    - `include`: regular expression to match for including classes. Defaults to `".*"`, i.e., all classes.
    - `exclude`: regular expression to match for excluding classes. Defaults to `None`.

    ##### Examples

    ```python
    >>> from pprint import pprint
    >>> pprint(find_classes(include='.*Stemmer', exclude='.*Lancaster.*'))
    [<class 'autogoal.contrib.nltk._generated.ISRIStemmer'>,
     <class 'autogoal.contrib.nltk._generated.PorterStemmer'>,
     <class 'autogoal.contrib.nltk._generated.RSLPStemmer'>,
     <class 'autogoal.contrib.nltk._generated.SnowballStemmer'>]

    ```
    """
    import autogoal.contrib.nltk._generated as module
    # from autogoal.contrib.nltk._builder import SklearnEstimator, SklearnTransformer

    return [
        c
        for n, c in inspect.getmembers(
            module,
            lambda c: inspect.isclass(c)
            and c.__module__ == 'autogoal.contrib.nltk._generated'
            # and issubclass(c, (SklearnEstimator, SklearnTransformer))
            and re.match(include, c.__name__)
            and (exclude is None or not re.match(exclude, c.__name__)),
        )
    ]
