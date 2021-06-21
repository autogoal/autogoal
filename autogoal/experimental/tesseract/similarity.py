from difflib import SequenceMatcher


def similarity(y, y_pred):
    return sum([SequenceMatcher(None, i, j).ratio() for i, j in zip(y, y_pred)])/len(y)

def similarityQuick(y, y_pred):
    return sum([SequenceMatcher(None, i, j).quick_ratio() for i, j in zip(y, y_pred)])/len(y)

def similarityRealQuick(y, y_pred):
    return sum([SequenceMatcher(None, i, j).real_quick_ratio() for i, j in zip(y, y_pred)])/len(y)


