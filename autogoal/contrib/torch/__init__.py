try:
    import torch

    # assert sklearn.__version__ == "0.22"
except:
    print("(!) Code in `autogoal.contrib.torch` requires `pytorch==0.1.4`.")
    print("(!) You can install it with `pip install autogoal[torch]`.")
    raise


from ._bert import BertEmbedding, BertTokenizeEmbedding
