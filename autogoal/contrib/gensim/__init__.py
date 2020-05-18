try:
    import gensim

    # assert gensim.__version__ == "3.8.1"
except:
    print("(!) Code in `autogoal.contrib.gensim` requires `gensim==3.8.1`.")
    print("(!) You can install it with `pip install autogoal[gensim]`.")
    raise


import inspect
import re


def find_classes(include=".*", exclude=None):
    """
    Returns the list of all `gensim` wrappers in `autogoal`.

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
    >>> pprint(find_classes())
    [<class 'autogoal.contrib.gensim._base.Word2VecEmbedding'>,
     <class 'autogoal.contrib.gensim._base.Word2VecEmbeddingSpanish'>]

    ```
    """
    from autogoal.contrib.gensim import _base as module

    def is_gensim_class(c):
        return (
            inspect.isclass(c)
            and c.__module__.startswith("autogoal.contrib.gensim._base")
            and re.match(include, c.__name__)
            and (exclude is None or not re.match(exclude, c.__name__))
            and hasattr(c, "run")
        )

    return [c for n, c in inspect.getmembers(module, is_gensim_class)]
