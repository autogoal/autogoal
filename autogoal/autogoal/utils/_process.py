from collections import deque
from itertools import chain
import linecache
import pickle
import warnings
from deprecated import deprecated
import psutil
import signal
import os
import logging
import platform
from numpy.core._exceptions import _ArrayMemoryError
import dill
from joblib import Parallel, delayed
import tracemalloc
import cloudpickle

from pathlib import Path
import sys
import gc
from collections.abc import Mapping, Container

try:
    import torch.multiprocessing as multiprocessing
except:
    import multiprocessing
if platform.system() == "Linux":
    import resource

TEMPORARY_DATA_PATH = Path.home() / ".autogoal" / "automl"


def ensure_temporary_data_path():
    global TEMPORARY_DATA_PATH
    os.makedirs(TEMPORARY_DATA_PATH, exist_ok=True)


def delete_temporary_data_path():
    os.remove(TEMPORARY_DATA_PATH)
    os.removedirs(TEMPORARY_DATA_PATH)


from autogoal.utils import Mb

logger = logging.getLogger("autogoal")

IS_MP_CUDA_INITIALIZED = False


def initialize_cuda_multiprocessing():
    global IS_MP_CUDA_INITIALIZED
    multiprocessing.set_start_method("spawn")
    IS_MP_CUDA_INITIALIZED = True


def is_cuda_multiprocessing_enabled():
    start_method = multiprocessing.get_start_method()
    return start_method == "spawn" or start_method == "forkserver"


class RestrictedWorker:
    def __init__(self, function, timeout: int, memory: int):
        self.function = function
        self.timeout = timeout
        self.memory = memory
        signal.signal(signal.SIGXCPU, alarm_handler)

    def _restrict(self):
        if platform.system() == "Linux":
            msoft, mhard = resource.getrlimit(resource.RLIMIT_DATA)
            csoft, chard = resource.getrlimit(resource.RLIMIT_CPU)
            used_memory = self.get_used_memory()

            if self.memory and self.memory > (used_memory + 500 * Mb):
                # memory may be restricted
                self.memory = min(self.memory, sys.maxsize)
                resource.setrlimit(resource.RLIMIT_DATA, (self.memory, mhard))
            else:
                warnings.warn("Cannot restrict memory")

            if self.timeout:
                # time may be restricted
                resource.setrlimit(resource.RLIMIT_CPU, (self.timeout, chard))
            else:
                warnings.warn("Cannot restrict cpu time")

    def _restricted_function(self, result_bucket, pipeline, *args, **kwargs):
        try:
            self._restrict()
            result = self.function(pipeline, *args, **kwargs)
            result_bucket["result"] = result
        except _ArrayMemoryError as e:
            result_bucket["result"] = _ArrayMemoryError(e.shape, e.dtype)
        except Exception as e:
            result_bucket["result"] = e

    def run_restricted(self, *args, **kwargs):
        """
        Executes a given function with restricted amount of
        CPU time and RAM memory usage
        """
        manager = multiprocessing.Manager()
        result_bucket = manager.dict()

        rprocess = multiprocessing.Process(
            target=self._restricted_function, args=[result_bucket, args, kwargs]
        )

        rprocess.start()
        rprocess.join(timeout=self.timeout)

        if rprocess.is_alive():
            rprocess.terminate()
            raise TimeoutError(
                f"Process took more than {self.timeout} seconds to complete and has been terminated."
            )

        result = result_bucket["result"]

        if isinstance(result, Exception):  # Exception ocurred
            raise result

        return result

    def get_used_memory(self):
        """
        returns the amount of memory being used by the current process
        """
        process = psutil.Process(os.getpid())
        return process.memory_info().rss

    def __call__(self, *args, **kwargs):
        return self.run_restricted(*args, **kwargs)


def alarm_handler(*args):
    raise TimeoutError("process %d got to time limit" % os.getpid())


class RestrictedWorkerByJoin(RestrictedWorker):
    def __init__(self, function, timeout: int, memory: int):
        if is_cuda_multiprocessing_enabled():
            self.p_function = cloudpickle.dumps(function)
        else:
            self.function = function
        self.timeout = timeout
        self.memory = memory

    def _restricted_function_with_serialization(
        self, result_bucket, pipeline, p_algorithms, p_input_types, *args, **kwargs
    ):
        try:
            self._restrict()
            function = cloudpickle.loads(self.p_function)
            pipeline.deserialize_inner_algorithms(p_algorithms)
            input_types = [pickle.loads(i) for i in p_input_types]
            pipeline.input_types = input_types

            result = function(pipeline, *args, **kwargs)
            result_bucket["result"] = result
        except _ArrayMemoryError as e:
            result_bucket["result"] = _ArrayMemoryError(e.shape, e.dtype)
        except MemoryError as e:
            result_bucket["result"] = MemoryError(f"Process exceeded memory limit of {self.memory/1024**2} MBs")
        except Exception as e:
            result_bucket["result"] = e

    def _restrict(self):
        if platform.system() == "Linux":
            _, mhard = resource.getrlimit(resource.RLIMIT_AS)
            used_memory = self.get_used_memory()

            if self.memory is None:
                return

            if self.memory > (used_memory + 50 * Mb):
                # memory may be restricted
                self.memory = min(self.memory, sys.maxsize)
                logger.info("💻 Restricting memory to %s" % self.memory)
                try:
                    resource.setrlimit(resource.RLIMIT_DATA, (self.memory, mhard))
                except Exception as e:
                    logger.info("💻 Failed to restrict memory to %s" % self.memory)
                    raise e
            else:
                raise Exception(
                    "Cannot restrict memory to %s < %i"
                    % (self.memory, used_memory + 50 * Mb)
                )

    def run_restricted(self, pipeline, *args, **kwargs):
        """
        Executes a given function with restricted amount of
        CPU time and RAM memory usage
        """
        manager = multiprocessing.Manager()
        result_bucket = manager.dict()

        if is_cuda_multiprocessing_enabled():
            p_input_types = [pickle.dumps(i) for i in pipeline.input_types]
            p_algorithms = pipeline.serialize_inner_algorithms()
            pipeline.algorithms = []
            rprocess = multiprocessing.Process(
                target=self._restricted_function_with_serialization,
                args=[result_bucket, pipeline, p_algorithms, p_input_types, *args],
                kwargs=kwargs,
            )
        else:
            rprocess = multiprocessing.Process(
                target=self._restricted_function,
                args=[result_bucket, pipeline, *args],
                kwargs=kwargs,
            )

        from torch.cuda import is_initialized

        print("CUDA Initialized:", is_initialized())

        rprocess.start()
        rprocess.join(self.timeout)

        if rprocess.exitcode == 0:
            result = result_bucket["result"]
        else:
            rprocess.terminate()
            raise TimeoutError(
                f"Exceded allowed time for execution. Any restricted function should end its excution in a timespan of {self.timeout} seconds."
            )

        if isinstance(result, Exception):  # Exception ocurred
            raise result

        print("after")
        return result


class RestrictedWorkerDiskSerializableByJoin(RestrictedWorkerByJoin):
    def __init__(self, function, timeout: int, memory: int):
        self.function = dill.dumps(function)
        self.timeout = timeout
        self.memory = memory

    def _restricted_function(self, result_bucket, *args, **kwargs):
        try:
            self._restrict()

            from autogoal.kb import Pipeline

            algorithms, types = Pipeline.load_algorithms(TEMPORARY_DATA_PATH)
            pipeline = Pipeline(algorithms, types)

            function = dill.loads(self.function)
            result = function(pipeline)
            result_bucket["result"] = result
        except _ArrayMemoryError as e:
            result_bucket["result"] = _ArrayMemoryError(e.shape, e.dtype)
        except Exception as e:
            result_bucket["result"] = e

    def run_restricted(self, pipeline, *args, **kwargs):
        """
        Executes a given function with restricted amount of
        CPU time and RAM memory usage
        """
        manager = multiprocessing.Manager()
        result_bucket = manager.dict()

        # ensure the directory to where the pipeline is going
        # to be exported exists
        ensure_temporary_data_path()

        global TEMPORARY_DATA_PATH
        pipeline.save_algorithms(TEMPORARY_DATA_PATH)

        rprocess = multiprocessing.Process(
            target=self._restricted_function,
            args=[result_bucket, TEMPORARY_DATA_PATH, *args],
            kwargs=kwargs,
        )

        rprocess.start()
        rprocess.join(self.timeout)

        if rprocess.exitcode == 0:
            result = result_bucket["result"]
        else:
            rprocess.terminate()
            raise TimeoutError(
                f"Exceded allowed time for execution. Any restricted function should end its excution in a timespan of {self.timeout} seconds."
            )

        if isinstance(result, Exception):  # Exception ocurred
            raise result

        # load trained pipeline
        from autogoal.kb import Pipeline

        algorithms, _ = Pipeline.load_algorithms(TEMPORARY_DATA_PATH)
        pipeline.algorithms = algorithms

        # delete all generated temp files
        delete_temporary_data_path()
        return result


class RestrictedWorkerWithState(RestrictedWorkerByJoin):
    def __init__(self, function, timeout: int, memory: int):
        self.function = function
        self.timeout = timeout
        self.memory = memory

    def _restricted_function(self, result_bucket, arguments_bucket, *args, **kwargs):
        try:
            instance = arguments_bucket["instance"]
            self._restrict()
            result = self.function(instance, *args, **kwargs)
            result_bucket["result"] = result
            result_bucket["instance"] = instance
        except _ArrayMemoryError as e:
            result_bucket["result"] = _ArrayMemoryError(e.shape, e.dtype)
        except Exception as e:
            result_bucket["result"] = e

    def run_restricted(self, instance, *args, **kwargs):
        """
        Executes a given function with restricted amount of
        CPU time and RAM memory usage
        """
        manager = multiprocessing.Manager()
        result_bucket = manager.dict()
        arguments_bucket = manager.dict(
            {
                "instance": instance,
            }
        )

        rprocess = multiprocessing.Process(
            target=self._restricted_function,
            args=[result_bucket, arguments_bucket, *args],
            kwargs=kwargs,
        )

        rprocess.start()
        rprocess.join(self.timeout)

        if rprocess.exitcode == 0:
            result = result_bucket.get("result"), result_bucket.get("instance")
        else:
            rprocess.kill()
            raise TimeoutError(
                f"Exceded allowed time for execution. Any restricted function should end its excution in a timespan of {self.timeout} seconds."
            )

        if isinstance(result, Exception):  # Exception ocurred
            raise result

        return result


def get_used_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**2)


def get_total_memory_size(o, handlers={}, verbose=False):
    """Returns the approximate memory footprint an object and all of its contents."""
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {
        tuple: iter,
        list: iter,
        deque: iter,
        dict: dict_handler,
        set: iter,
        frozenset: iter,
        Mapping: dict_handler,
        Container: iter,
    }
    all_handlers.update(handlers)  # user-defined handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = sys.getsizeof(0)  # estimate sizeof(int) as a default size

    def size(o):
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = sys.getsizeof(o, default_size)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(size, handler(o)))
                break
        return s

    total = size(o)
    if verbose:
        print(f"Total memory: {total/1024**2} MB")
    return total


def print_biggest_memory_block_traceback():
    import tracemalloc

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("traceback")

    # pick the biggest memory block
    stat = top_stats[0]
    print("%s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))
    for line in stat.traceback.format():
        print(line)


def take_memory_snapshot(key_type="lineno", limit=20):
    tracemalloc.take_snapshot()
    snapshot = tracemalloc.take_snapshot()

    snapshot = snapshot.filter_traces(
        (
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
        )
    )
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print(
            "#%s: %s:%s: %.1f KiB"
            % (index, frame.filename, frame.lineno, stat.size / 1024)
        )
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print("    %s" % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def handler(signum, frame):
    raise TimeoutError()


def restrict(memory, timeout):
    if platform.system() == "Linux":
        _, mhard = resource.getrlimit(resource.RLIMIT_AS)
        used_memory = get_used_memory()
        if memory is None:
            return
        if memory > (used_memory + 200 * Mb):
            # memory may be restricted
            memory = min(memory, sys.maxsize)
            logger.info("💻 Restricting memory to %s" % memory)
            try:
                resource.setrlimit(resource.RLIMIT_AS, (memory, mhard))
            except Exception as e:
                logger.info("💻 Failed to restrict memory to %s" % memory)
                raise e
        else:
            raise Exception(
                "Cannot restrict memory to %s < %i" % (memory, used_memory + 200 * Mb)
            )

        # signal.signal(signal.SIGALRM, handler)
        # signal.alarm(timeout)  # Number of seconds before alarm is raised


def get_current_memory_limit():
    if platform.system() == "Linux":
        _, mhard = resource.getrlimit(resource.RLIMIT_AS)
        return mhard
    return None


def clear_cuda_memory():
    try:
        import torch

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except:
        pass


def mock_function(*args, **kwargs):
    """
    A mock function that does minimal work.
    Its primary purpose is to allow the usage of multiple jobs in joblib
    to enable setting a timeout for the primary task.
    """
    pass


def restricted_function(memory, timeout, serialized_input_types, *args, **kwargs):
    try:
        restrict(int(memory), int(timeout))
        print("Limit in thread", resource.getrlimit(resource.RLIMIT_AS))

        deserialized_input_types = [
            cloudpickle.loads(i) for i in serialized_input_types
        ]
        # pipeline.input_types = deserialized_input_types
        # return function(pipeline, *args, **kwargs)
    except _ArrayMemoryError as e:
        clear_cuda_memory()
        print(
            f"Limit at the end {resource.getrlimit(resource.RLIMIT_AS)}\ncurrent memory usage {get_used_memory()} MB"
        )
        raise _ArrayMemoryError(e.shape, e.dtype)
    except TimeoutError as e:
        clear_cuda_memory()
        print(
            f"Limit at the end {resource.getrlimit(resource.RLIMIT_AS)}\ncurrent memory usage {get_used_memory()} MB"
        )
        raise TimeoutError(
            f"Exceded allowed time for execution. Any restricted function should end its excution in a timespan of {self.timeout} seconds."
        )
    except Exception as e:
        clear_cuda_memory()
        print(
            f"Limit at the end {resource.getrlimit(resource.RLIMIT_AS)}\ncurrent memory usage {get_used_memory()} MB"
        )
        if e.__class__.__name__ == "TimeoutError":
            raise TimeoutError(
                f"Exceded allowed time for execution. Any restricted function should end its excution in a timespan of {self.timeout} seconds."
            )
        if e.__class__.__name__ == "MemoryError":
            raise MemoryError(
                f"Exceded allowed memory for execution. Any restricted function should at must use {self.memory} bytes."
            )
        raise e

@deprecated(reason="Use RestrictedWorkerByJoin instead. JobLibRestrictedWorkerByJoin does not support CUDA multiprocessing.")
class JobLibRestrictedWorkerByJoin:
    def __init__(self, function, timeout: int, memory: int):
        self.function = function
        self.timeout = timeout
        self.memory = memory
        self.overall_memory_limit = get_current_memory_limit()

    def reset_memory_limit(self):
        gc.collect()
        if self.overall_memory_limit is not None:
            resource.setrlimit(
                resource.RLIMIT_AS,
                (self.overall_memory_limit, self.overall_memory_limit),
            )

    def run_restricted(self, pipeline, *args, **kwargs):
        """
        Executes a given function with restricted amount of
        CPU time and RAM memory usage, alongside a mock function
        to enable joblib's timeout feature.
        """
        try:

            from autogoal.kb import Pipeline

            input_types = pipeline.input_types
            serialized_input_types = [cloudpickle.dumps(i) for i in input_types]

            with Parallel(n_jobs=2, timeout=self.timeout, backend="loky") as parallel:
                result = parallel(
                    [
                        delayed(restricted_function)(
                            self.memory,
                            self.timeout,
                            # self.function,
                            # pipeline,
                            serialized_input_types,
                            *args,
                            **kwargs,
                        ),
                    ]
                )
            return result[0]
        except _ArrayMemoryError as e:
            clear_cuda_memory()
            self.reset_memory_limit()
            print(
                f"Limit at the end {resource.getrlimit(resource.RLIMIT_AS)}\ncurrent memory usage {get_used_memory()} MB"
            )
            raise _ArrayMemoryError(e.shape, e.dtype)
        except TimeoutError as e:
            clear_cuda_memory()
            self.reset_memory_limit()
            print(
                f"Limit at the end {resource.getrlimit(resource.RLIMIT_AS)}\ncurrent memory usage {get_used_memory()} MB"
            )
            raise TimeoutError(
                f"Exceded allowed time for execution. Any restricted function should end its excution in a timespan of {self.timeout} seconds."
            )
        except Exception as e:
            clear_cuda_memory()
            self.reset_memory_limit()
            print(f"Error {e}")
            print(
                f"Limit at the end {resource.getrlimit(resource.RLIMIT_AS)}\ncurrent memory usage {get_used_memory()} MB"
            )

            if e.__class__.__name__ == "TimeoutError":
                raise TimeoutError(
                    f"Exceded allowed time for execution. Any restricted function should end its excution in a timespan of {self.timeout} seconds."
                )

            if e.__class__.__name__ == "MemoryError":
                raise MemoryError(
                    f"Exceded allowed memory for execution. Any restricted function should at must use {self.memory} bytes."
                )
            raise e

    def __call__(self, *args, **kwargs):
        return self.run_restricted(*args, **kwargs)
