import signal
import resource
import psutil
import warnings
import os

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
    
    def _run_for(self, function, *args, **kwargs):
        def signal_handler(signum, frame):
            raise Exception("%s reached time limit (%d)" %(function.__name__, self.time_limit))
        
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(self.time_limit)
        
        return function(*args, **kwargs)
    
    def get_used_memory(self):
        """
        returns the amount of memory being used by the current process
        """
        process = psutil.Process(os.getpid())
        return process.memory_info().rss
    
    def run_restricted(self, function, *args, **kwargs):
        """
        Executes a given function with restricted amount of
        CPU time and RAM memory usage
        """
        self._restrict_memory(self.memory_limit)
        try:
            return self._run_for(function, *args, **kwargs)
        except MemoryError as e:
            # restore original memory limit
            self._restrict_memory(self.original_limit)
            raise MemoryError("Memory error found for function %s" %function.__name__)
            
        except Exception as e:
            # restore original memory limit
            self._restrict_memory(self.original_limit)
            raise e