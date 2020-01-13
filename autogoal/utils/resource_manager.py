import signal
import resource
import psutil
import warnings
import os
import multiprocessing

class ResourceManager:
    """
    Resource manager class.
    
    `params:`\n
        time_limit:int (max amount of seconds a function can run for)
        ram_limit:int (max amount of memory bytes a function can use)
    
    `defaults:`:
        time_limit: 5 minutes
        ram_limit: 4 gigabytes
        
    Notes: 
        - Only one function may be restricted at the same time.
          Upcoming updates will fix this matter.
          
        - Memory restriction is issued upon the process's heap memory size and not 
          over total address space in order to get a better estimation of the used memory.
    
    """
    def __init__(self, time_limit:int = 360, memory_limit:int = 4294967296):
        self.set_time_limit(time_limit)
        self.set_memory_limit(memory_limit)
     
    def set_memory_limit(self, limit):
        """
        set memory limit for future restricted functions
        
        If memory limit is higher or equal than the current OS limit
        then it won't be changed
        """
        soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
        self.original_limit = (soft, hard)
        self.memory_limit = None
        used_memory = self.get_used_memory()
        
        if limit <= (used_memory + 500000000):
            warnings.warn("Especified memory limit is too close to the used amount. Will not be taken into account.")
            return
        
        if soft == -1 or limit < soft:
            self.memory_limit = (limit, hard)
        else:
            warnings.warn("Especified memory limit is higher than OS limit. Will not be taken into account.")
    
    def set_time_limit(self, limit):
        self.time_limit = limit
        
    def _restrict_memory(self, memory_amount):
        if memory_amount:
            _, hard = self.original_limit
            limit, _ = memory_amount
            resource.setrlimit(resource.RLIMIT_DATA, (limit, hard))
    
    def _unrestrict_memory(self):
        self._restrict_memory(self.original_limit)
    
    def _run_for(self, function, *args, **kwargs):
        def signal_handler(*args):
            raise TimeoutError()
        
        try:
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(self.time_limit)
            
            result = function(*args, **kwargs)
            signal.alarm(0) #cancel the alarm
            return result
        except Exception as e:
            raise e
    
    def get_used_memory(self):
        """
        returns the amount of memory being used by the current process
        """
        process = psutil.Process(os.getpid())
        return process.memory_info().rss
    
    def _restricted_function(self, result_bucket, function, args, kwargs):
        try:
            self._restrict_memory(self.memory_limit)
            result = self._run_for(function, *args, **kwargs)
            result_bucket.put(result)
        except Exception as e:
            result_bucket.put(e)
    
    def run_restricted(self, function, *args, **kwargs):
        """
        Executes a given function with restricted amount of
        CPU time and RAM memory usage
        """
        try:
            result_bucket = multiprocessing.Queue()
            
            rprocess = multiprocessing.Process(target=self._restricted_function,
                                               args=[result_bucket, function, args, kwargs])

            rprocess.start()
            rprocess.join()
            
            result = result_bucket.get()
            if isinstance(result, Exception): #Exception ocurred
                raise result
            return result
                    
        except MemoryError as e:
            raise MemoryError("Memory error found for function %s" %function.__name__)
            
        except TimeoutError as e:
            raise TimeoutError("%s reached time limit (%d)" %(function.__name__, self.time_limit))
        
        except Exception as e:
            raise e