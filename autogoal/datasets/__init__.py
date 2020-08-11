import shutil
import json
import requests
import os
from tqdm import tqdm

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
    PosixPath('/home/coder/autogoal/autogoal/datasets/data/movie_reviews')

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

    download_and_save(url, path, True)

    if unpackit:
        unpack(fname)


def download_and_save(url, path: Path, overwrite=False, data_length=None):
    stream = requests.get(url, stream=True)
    total_size = data_length or int(stream.headers.get('content-length', 0))

    if path.exists() and not overwrite:
        return False

    try:
        with path.open("wb") as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                for data in stream.iter_content(32*1024):
                    f.write(data)
                    pbar.update(len(data))

        return True
    except:
        path.unlink()
        raise
