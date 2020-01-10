try:
    import sklearn

    assert sklearn.__version__ == "0.22"
except:
    print("(!) Code in `autogoal.contrib.sklearn` requires `sklearn==0.22`.")
    print("(!) You can install it with `pip install autogoal[sklearn]`.")
    raise


from ._pipeline import SklearnClassifier
