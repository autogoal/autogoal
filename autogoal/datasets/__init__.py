from pathlib import Path


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
