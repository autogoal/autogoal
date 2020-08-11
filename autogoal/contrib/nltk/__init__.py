try:
    import nltk
except:
    print("(!) Code in `autogoal.contrib.nltk` requires `nltk`.")
    print("(!) You can install it with `pip install autogoal[nltk]`.")
    raise

from autogoal.contrib.nltk._generated import *
