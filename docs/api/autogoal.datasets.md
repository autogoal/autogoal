# `autogoal.datasets`

## Submodules

* [autogoal.datasets.movie_reviews](/api/autogoal.datasets.movie_reviews/)

## Classes

### `Path`

> [📝](/usr/local/lib/python3.6/pathlib.py#L984)
> `Path(self, *args, **kwargs)`

PurePath subclass that can make system calls.

    Path represents a filesystem path but unlike PurePath, also offers
    methods to do system calls on path objects. Depending on your system,
    instantiating a Path will return either a PosixPath or a WindowsPath
    object. You can also instantiate a PosixPath or WindowsPath directly,
    but cannot instantiate a WindowsPath on a POSIX system or vice versa.

#### `cwd`

> [📝](/usr/local/lib/python3.6/pathlib.py#L1050)
> `cwd()`

Return a new path pointing to the current working directory
        (as returned by os.getcwd()).

#### `home`

> [📝](/usr/local/lib/python3.6/pathlib.py#L1057)
> `home()`

Return a new path pointing to the user's home directory (as
        returned by os.path.expanduser('~')).


## Functions

### `datapath`

> [📝](https://github.com/sestevez/autogoal/blob/master/autogoal/datasets/__init__.py#L13)
> `datapath(path)`


Returns a `Path` object that points to the dataset path
where `path` is located.

##### Examples

```python
>>> datapath("movie_reviews")
PosixPath('/code/autogoal/datasets/data/movie_reviews')

```

### `download`

> [📝](https://github.com/sestevez/autogoal/blob/master/autogoal/datasets/__init__.py#L41)
> `download(dataset, unpackit=True)`


!!! warning
    This class has no docstrings.

### `pack`

> [📝](https://github.com/sestevez/autogoal/blob/master/autogoal/datasets/__init__.py#L29)
> `pack(folder)`


!!! warning
    This class has no docstrings.

### `unpack`

> [📝](https://github.com/sestevez/autogoal/blob/master/autogoal/datasets/__init__.py#L35)
> `unpack(zipfile)`


!!! warning
    This class has no docstrings.

