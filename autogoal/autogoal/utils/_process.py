import warnings
import psutil
import signal
import os
import logging
import platform
from numpy.core._exceptions import _ArrayMemoryError
import dill
# from autogoal.kb import Pipeline
from pathlib import Path
import sys

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
    try:
        global IS_MP_CUDA_INITIALIZED
        if not IS_MP_CUDA_INITIALIZED:
            multiprocessing.set_start_method('forkserver', force=True)
            print("initialized multiprocessing")
            IS_MP_CUDA_INITIALIZED = True
    except:
        return

def is_cuda_multiprocessing_enabled():
    start_method = multiprocessing.get_start_method()
    return start_method == 'spawn' or start_method == 'forkserver'

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

    def _restricted_function(self, result_bucket, *args, **kwargs):
        try:
            self._restrict()
            result = self.function(*args, **kwargs)
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
        self.function = function
        self.timeout = timeout
        self.memory = memory

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
                raise Exception (
                    "Cannot restrict memory to %s < %i"
                    % (self.memory, used_memory + 50 * Mb)
                )

    def run_restricted(self, *args, **kwargs):
        """
        Executes a given function with restricted amount of
        CPU time and RAM memory usage
        """
        manager = multiprocessing.Manager()
        result_bucket = manager.dict()

        rprocess = multiprocessing.Process(
            target=self._restricted_function, args=[result_bucket, *args], kwargs=kwargs
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
            target=self._restricted_function, args=[result_bucket, TEMPORARY_DATA_PATH, *args], kwargs=kwargs
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


import joblib
from joblib import Parallel, delayed, wrap_non_picklable_objects

def get_used_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

def handler(signum, frame):
    raise TimeoutError()

def restrict(memory, timeout):
    if platform.system() == "Linux":
        _, mhard = resource.getrlimit(resource.RLIMIT_AS)
        used_memory = get_used_memory()
        if memory is None:
            return
        if memory > (used_memory + 50 * Mb):
            # memory may be restricted
            memory = min(memory, sys.maxsize)
            logger.info("💻 Restricting memory to %s" % memory)
            try: 
                resource.setrlimit(resource.RLIMIT_DATA, (memory, mhard))
            except Exception as e:
                logger.info("💻 Failed to restrict memory to %s" % memory)
                raise e
        else:
            raise Exception (
                "Cannot restrict memory to %s < %i"
                % (memory, used_memory + 50 * Mb)
            )
        
        # signal.signal(signal.SIGALRM, handler)
        # signal.alarm(timeout)  # Number of seconds before alarm is raised

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

def restricted_function(memory, timeout, function, *args, **kwargs):
    try:
        restrict(int(memory), int(timeout))
        clear_cuda_memory()
        return function(*args, **kwargs)
    except _ArrayMemoryError as e:
        clear_cuda_memory()
        raise _ArrayMemoryError(e.shape, e.dtype)
    except TimeoutError as e:
        clear_cuda_memory()
        raise TimeoutError(
                f"Exceded allowed time for execution. Any restricted function should end its excution in a timespan of {timeout} seconds."
            )
    except Exception as e:
        clear_cuda_memory()
        raise e

class JobLibRestrictedWorkerByJoin:
    def __init__(self, function, timeout: int, memory: int):
        self.function = function
        self.timeout = timeout
        self.memory = memory
        
    def run_restricted(self, pipeline, *args, **kwargs):
        """
        Executes a given function with restricted amount of
        CPU time and RAM memory usage, alongside a mock function
        to enable joblib's timeout feature.
        """
        try:
            with Parallel(n_jobs=2, backend='threading', timeout=self.timeout) as parallel:
                result = parallel([
                    delayed(restricted_function)(self.memory, self.timeout, self.function, pipeline, *args, **kwargs),
                    delayed(mock_function)()
                ])
            return result[0]
        except _ArrayMemoryError as e:
            clear_cuda_memory()
            raise _ArrayMemoryError(e.shape, e.dtype)
        except TimeoutError as e:
            clear_cuda_memory()
            raise TimeoutError(
                    f"Exceded allowed time for execution. Any restricted function should end its excution in a timespan of {self.timeout} seconds."
                )
        except Exception as e:
            clear_cuda_memory()
            if (e.__class__.__name__ == "TimeoutError"):
                raise TimeoutError(
                    f"Exceded allowed time for execution. Any restricted function should end its excution in a timespan of {self.timeout} seconds."
                )
            raise e

    def __call__(self, *args, **kwargs):
        return self.run_restricted(*args, **kwargs)