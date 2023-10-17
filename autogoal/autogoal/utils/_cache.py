import mmap
import functools


import pickle, json, csv, os, shutil


class PersistentDict(dict):
    """
    Persistent dictionary with an API compatible with shelve and anydbm.

    Taken from <https://code.activestate.com/recipes/576642/>

    The dict is kept in memory, so the dictionary operations run as fast as
    a regular dictionary.

    Write to disk is delayed until close or sync (similar to gdbm's fast mode).

    Input file format is automatically discovered.
    Output file format is selectable between pickle, json, and csv.
    All three serialization formats are backed by fast C implementations.

    """

    def __init__(self, filename, flag="c", mode=None, format="pickle", *args, **kwds):
        self.flag = flag  # r=readonly, c=create, or n=new
        self.mode = mode  # None or an octal triple like 0644
        self.format = format  # 'csv', 'json', or 'pickle'
        self.filename = filename
        if flag != "n" and os.access(filename, os.R_OK):
            fileobj = open(filename, "rb" if format == "pickle" else "r")
            with fileobj:
                self.load(fileobj)
        dict.__init__(self, *args, **kwds)

    def sync(self):
        "Write dict to disk"
        if self.flag == "r":
            return
        filename = self.filename
        tempname = filename + ".tmp"
        fileobj = open(tempname, "wb" if self.format == "pickle" else "w")
        try:
            self.dump(fileobj)
        except Exception:
            os.remove(tempname)
            raise
        finally:
            fileobj.close()
        shutil.move(tempname, self.filename)  # atomic commit
        if self.mode is not None:
            os.chmod(self.filename, self.mode)

    def close(self):
        self.sync()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def dump(self, fileobj):
        if self.format == "csv":
            csv.writer(fileobj).writerows(self.items())
        elif self.format == "json":
            json.dump(self, fileobj, separators=(",", ":"))
        elif self.format == "pickle":
            pickle.dump(dict(self), fileobj, 2)
        else:
            raise NotImplementedError("Unknown format: " + repr(self.format))

    def load(self, fileobj):
        # try formats from most restrictive to least restrictive
        for loader in (pickle.load, json.load, csv.reader):
            fileobj.seek(0)
            try:
                return self.update(loader(fileobj))
            except Exception:
                pass
        raise ValueError("File not in a supported format")


from pathlib import Path


class CacheManager:
    _instance = None

    def __init__(self):
        self.cache = PersistentDict(str(Path(__file__).parent / "cache.pickle"))

    @staticmethod
    def get(name: str, func):
        instance = CacheManager.instance()

        if name not in instance.cache:
            print("Creating cached object '%s'" % name)
            instance.cache[name] = func()
            instance.cache.sync()

        return instance.cache[name]

    @staticmethod
    def instance() -> "CacheManager":
        if CacheManager._instance is None:
            CacheManager._instance = CacheManager()

        return CacheManager._instance


def cached_run(func):
    @functools.wraps(func)
    def run(self, input):
        if not hasattr(input, "__cached_id__"):
            return func(input)

        cached_id = input.__cached_id__

    return run
