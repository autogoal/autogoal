import signal
import psutil
import warnings
import os
import multiprocessing
import platform

if platform.system() == "Linux":
    import resource


class ResourceManager:
    """
    Resource manager class.

    ##### Parameters

    - `time_limit: int`: maximum amount of seconds a function can run for (default = `5 minutes`).
    - `ram_limit: int`: maximum amount of memory bytes a function can use (default = `4 GB`).

    ##### Notes

    - Only one function may be restricted at the same time.
      Upcoming updates will fix this matter.
    - Memory restriction is issued upon the process's heap memory size and not
      over total address space in order to get a better estimation of the used memory.

    """

    def __init__(self, time_limit: int = 300, memory_limit: int = 4294967296):
        self.set_time_limit(time_limit)
        self.set_memory_limit(memory_limit)
        signal.signal(signal.SIGXCPU, alarm_handler)

    def set_memory_limit(self, limit):
        """
        Set the memory limit for future restricted functions.

        If memory limit is higher or equal than the current OS limit
        then it won't be changed.
        """
        soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
        self.original_limit = (soft, hard)
        self.memory_limit = None
        used_memory = self.get_used_memory()

        if limit <= (used_memory + 500 * 1024 ** 2):
            warnings.warn(
                "Especified memory limit is too close to the used amount. Will not be taken into account."
            )
            return

        if soft == -1 or limit < soft:
            self.memory_limit = (limit, hard)
        else:
            warnings.warn(
                "Especified memory limit is higher than OS limit. Will not be taken into account."
            )

    def set_time_limit(self, limit):
        self.time_limit = limit

    def _restrict(self, memory_amount):
        if memory_amount:
            _, hard = self.original_limit
            limit, _ = memory_amount
            resource.setrlimit(resource.RLIMIT_DATA, (limit, hard))
            resource.setrlimit(resource.RLIMIT_CPU, (self.time_limit, hard))

    def _unrestrict_memory(self):
        self._restrict(self.original_limit)

    def _run_for(self, function, *args, **kwargs):
        def signal_handler(*args):
            raise TimeoutError()

        try:
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(self.time_limit)

            result = function(*args, **kwargs)
            signal.alarm(0)  # cancel the alarm
            return result
        except Exception as e:
            # signal.alarm(0) #cancel the alarm
            raise e

    def get_used_memory(self):
        """
        Returns the amount of memory being used by the current process.
        """
        process = psutil.Process(os.getpid())
        return process.memory_info().rss

    def _restricted_function(self, result_bucket, function, args, kwargs):
        try:
            self._restrict(self.memory_limit)
            result = function(*args, **kwargs)
            result_bucket["result"] = result
        except Exception as e:
            result_bucket["result"] = e

    def run_restricted(self, function, *args, **kwargs):
        """
        Executes a given function with restricted amount of
        CPU time and RAM memory usage.
        """
        manager = multiprocessing.Manager()
        result_bucket = manager.dict()

        rprocess = multiprocessing.Process(
            target=self._restricted_function,
            args=[result_bucket, function, args, kwargs],
        )

        rprocess.start()
        # print("started process:", rprocess.pid)
        rprocess.join()
        # print("ended process:", rprocess.pid)
        result = result_bucket["result"]
        if isinstance(result, Exception):  # Exception ocurred
            raise result
        return result


def alarm_handler(*args):
    raise TimeoutError("process %d got to time limit" % os.getpid())
