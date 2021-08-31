# Gelin Eguinosa Rosique

try:
    import fasttext
except:
    raise ImportError("autogoal.experimental.translator requires 'fasttext'")

try:
    import torch
except:
    raise ImportError("autogoal.experimental.translator requires 'torch'")

try:
    import sentencepiece
except:
    raise ImportError("autogoal.experimental.translator requires 'sentencepiece'")

try:
    import transformers
except:
    raise ImportError("autogoal.experimental.translator requires 'transformers'")

try:
    import sentence_transformers
except:
    raise ImportError("autogoal.experimental.translator requires 'sentence_transformers'")
