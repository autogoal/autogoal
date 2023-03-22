import os
from pathlib import Path

CONTRIB_NAME = "gensim"
DATA_PATH = Path.home() / ".autogoal" / "contrib" / CONTRIB_NAME / "data"

# ensure data path directory creation
try:
    os.makedirs(DATA_PATH)
except IOError as ex:
    # directory already exists
    pass


def load_data(path, name):
    use_data(path)
    try:
        import gensim
        import gensim.downloader as api

        api.load(name)
    except:
        pass


def use_data(path):
    os.environ["GENSIM_DATA_DIR"] = str(path)
