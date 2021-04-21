import os
from pathlib import Path

DATA_PATH = Path.home() / ".autogoal" / "contrib" / "transformers"

# Setting up download location
os.environ["TRANSFORMERS_CACHE"] = str(DATA_PATH)


try:
    import torch
    import transformers

    # assert sklearn.__version__ == "0.22"
except:
    print(
        "(!) Code in `autogoal.contrib.transformers` requires `pytorch==0.1.4` and `transformers==2.3.0`."
    )
    print("(!) You can install it with `pip install autogoal[transformers]`.")
    raise


from ._bert import BertEmbedding, BertTokenizeEmbedding


def download():
    BertEmbedding.download()
    return True


def status():
    from autogoal.contrib import ContribStatus

    try:
        BertEmbedding.check_files()
    except OSError:
        return ContribStatus.RequiresDownload

    return ContribStatus.Ready
