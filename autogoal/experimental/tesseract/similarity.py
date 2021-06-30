from difflib import SequenceMatcher


def similarity(y, y_pred):
    return sum([_selectSimilarityType(i, j).ratio() for i, j in zip(y, y_pred)])/len(y)

def similarityQuick(y, y_pred):
    return sum([_selectSimilarityType(i, j).quick_ratio() for i, j in zip(y, y_pred)])/len(y)

def similarityRealQuick(y, y_pred):
    return sum([_selectSimilarityType(i, j).real_quick_ratio() for i, j in zip(y, y_pred)])/len(y)

def _selectSimilarityType(x, y):
    if(isinstance(y,dict)):
       return _similarityBetweenTextandDict(x, y)
    else:
        return _similarityBetweenTexts(x, y)

def _similarityBetweenTexts(text1, text2):
    return SequenceMatcher(None, text1, text2)

def _similarityBetweenTextandDict(text, dict):
    if('text' in dict.keys()):
        newText=' '.join(str(e) for e in dict['text'])
    elif('char' in dict.keys()):
        newText=''.join(str(e) for e in dict['char'])
    else:
        newText=''
    
    return SequenceMatcher(None, newText, text)