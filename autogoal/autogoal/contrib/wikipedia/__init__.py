try:
    import wikipedia

    # assert wikipedia.__version__ == "1.4.0"
except:
    print("(!) Code in `autogoal.contrib.wikipedia` requires `wikipedia==1.4.0`.")
    print("(!) You can install it with `pip install autogoal[wikipedia]`.")
    raise


from ._base import (
    WikipediaContainsWord,
    WikipediaContainsWordSpanish,
    WikipediaSummary,
    WikipediaSummarySpanish,
)
