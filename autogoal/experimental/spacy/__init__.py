try:
    import spacy
except:
    print("(!) Code in `autogoal.contrib.spacy` requires `spacy =^3`.")
    print("(!) You can install it with `pip install autogoal[spacy]`.")
    raise ImportError()


from autogoal.contrib.spacy._manual import SpacyTagger, SpacyNER
