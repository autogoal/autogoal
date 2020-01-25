import multiprocessing
import warnings
import resource
import psutil
import signal
import os

class RestrictedWorker():
    def __init__(self, function,  timeout:int = 360, memory:int=4 * 1024 ** 3):
        self.function = function
        self.timeout = timeout
        self.memory = memory
        signal.signal(signal.SIGXCPU, alarm_handler)
               
    def _restrict(self):
        msoft, mhard = resource.getrlimit(resource.RLIMIT_DATA)
        csoft, chard = resource.getrlimit(resource.RLIMIT_CPU)
        used_memory = self.get_used_memory()
        
        if self.memory and self.memory > (used_memory + 500000000):
            #memory may be restricted
            resource.setrlimit(resource.RLIMIT_DATA, (self.memory, mhard))
        else:
            warnings.warn("Cannot restrict memory")
            
        if self.timeout:
            #time may be restricted
            resource.setrlimit(resource.RLIMIT_CPU, (self.timeout, chard))
        else:
            warnings.warn("Cannot restrict cpu time")
    
    def _restricted_function(self, result_bucket, args, kwargs):
        try:
            self._restrict()
            result = self.function(*args, **kwargs)
            result_bucket["result"] = result
        except MemoryError:
            result_bucket["result"] = Exception("Memory error")
        except Exception as e:
            result_bucket["result"] = Exception(str(e))
    
    def run_restricted(self, *args, **kwargs):
        """
        Executes a given function with restricted amount of
        CPU time and RAM memory usage
        """
        manager = multiprocessing.Manager()
        result_bucket = manager.dict()
        
        rprocess = multiprocessing.Process(target=self._restricted_function,
                                            args=[result_bucket, args, kwargs])

        rprocess.start()
        # print("started process:", rprocess.pid)
        rprocess.join()
        # print("ended process:", rprocess.pid)
        result = result_bucket["result"]
        if isinstance(result, Exception): #Exception ocurred
            # print("exception ocurred %s:" %result)
            raise result
        # print("result:%g" %result)
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
    raise TimeoutError("process %d got to time limit" %os.getpid())