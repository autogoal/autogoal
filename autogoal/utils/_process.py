import multiprocessing
import warnings
import resource
import psutil
import signal
import os
import sys
import traceback

from autogoal.utils import Mb


class RestrictedWorker:
    def __init__(self, function, timeout: int, memory: int):
        self.function = function
        self.timeout = timeout
        self.memory = memory
        signal.signal(signal.SIGXCPU, alarm_handler)

    def _restrict(self):
        msoft, mhard = resource.getrlimit(resource.RLIMIT_DATA)
        csoft, chard = resource.getrlimit(resource.RLIMIT_CPU)
        used_memory = self.get_used_memory()

        if self.memory and self.memory > (used_memory + 500 * Mb):
            # memory may be restricted
            resource.setrlimit(resource.RLIMIT_DATA, (self.memory, mhard))
        else:
            warnings.warn("Cannot restrict memory")

        if self.timeout:
            # time may be restricted
            resource.setrlimit(resource.RLIMIT_CPU, (self.timeout, chard))
        else:
            warnings.warn("Cannot restrict cpu time")

    def _restricted_function(self, result_bucket, args, kwargs):
        try:
            self._restrict()
            result = self.function(*args, **kwargs)
            result_bucket["result"] = result
        except Exception as e:
            msg = "{}\n\nOriginal {}".format(e, traceback.format_exc())
            result_bucket["result"] = e.__class__(msg)

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
        rprocess.join()
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
        _, mhard = resource.getrlimit(resource.RLIMIT_AS)
        used_memory = self.get_used_memory()

        if self.memory is None:
            return

        if self.memory > (used_memory + 50 * Mb):
            # memory may be restricted
            print("Restricting memory to %s" % self.memory, flush=True)
            resource.setrlimit(resource.RLIMIT_DATA, (self.memory, mhard))
        else:
            raise ValueError("Cannot restrict memory to %s < %i" % (self.memory, used_memory + 50 * Mb))

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
        rprocess.join(self.timeout)

        if rprocess.exitcode == 0:
            result = result_bucket["result"]
        else:
            rprocess.terminate()
            raise TimeoutError()

        if isinstance(result, Exception):  # Exception ocurred
            raise result

        return result
