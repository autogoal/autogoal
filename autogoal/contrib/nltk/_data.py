import os
import nltk
from pathlib import Path

CONTRIB_NAME = "nltk"
DATA_PATH = Path.home() / ".autogoal" / "contrib" / CONTRIB_NAME / "data"

# ensure data path directory creation
try:
    os.makedirs(DATA_PATH)
except IOError as ex:
    # directory already exists
    pass


def load_data(path):
    nltk.download(download_dir=path, quiet=True)
    use_data(path)


def use_data(path):
    nltk.data.path.append(path)
