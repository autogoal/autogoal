try:
    import sklearn

    major, minor, *rest = sklearn.__version__.split(".")
    assert int(major) == 0 and int(minor) >= 22
except:
    print("(!) Code in `autogoal.contrib.sklearn` requires `sklearn=^0.22`.")
    print("(!) You can install it with `pip install autogoal[sklearn]`.")
    raise


# from ._pipeline import SklearnClassifier
