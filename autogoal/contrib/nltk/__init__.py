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
    import autogoal.contrib.nltk._generated as generated
    import autogoal.contrib.nltk._manual as manual

    def is_nltk_class(c):
        return (
            inspect.isclass(c)
            and (
                c.__module__.startswith("autogoal.contrib.nltk._generated")
                or c.__module__.startswith("autogoal.contrib.nltk._manual")
            )
            and re.match(include, c.__name__)
            and (exclude is None or not re.match(exclude, c.__name__))
            and hasattr(c, "run")
        )

    return [
        c
        for n, c in inspect.getmembers(generated, is_nltk_class)
        + inspect.getmembers(manual, is_nltk_class)
    ]
