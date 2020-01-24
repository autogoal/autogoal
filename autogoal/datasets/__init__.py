import shutil
import json
import requests
import os

from pathlib import Path

DATASETS_METADATA = (
    "https://raw.githubusercontent.com/autogoal/datasets/master/datasets.json"
)


def datapath(path: str) -> Path:
    """
    Returns a `Path` object that points to the dataset path
    where `path` is located.

    ##### Examples

    ```python
    >>> datapath("movie_reviews")
    PosixPath('/code/autogoal/datasets/data/movie_reviews')

    ```
    """
    return Path(__file__).parent / "data" / path


def pack(folder: str):
    filename = datapath(folder)
    rootdir = datapath(folder)
    shutil.make_archive(filename, "zip", root_dir=rootdir)


def unpack(zipfile: str):
    filename = datapath(zipfile)
    rootdir = datapath(zipfile[:-4])
    shutil.unpack_archive(filename, extract_dir=rootdir, format="zip")


def download(dataset: str, unpackit: bool = True):
    fname = f"{dataset}.zip"
    path = datapath(fname)

    if path.exists():
        return

    datasets = requests.get(DATASETS_METADATA).json()
    url = datasets[dataset]

    with path.open("wb") as fp:
        stream = requests.get(url)
        fp.write(stream.content)

    if unpackit:
        unpack(fname)
