try:
    import fasttext
except:
    print("(!) Code in `autogoal.contrib.fasttext` requires `fasttext`.")
    raise ImportError()