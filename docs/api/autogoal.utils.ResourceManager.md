# `autogoal.utils.ResourceManager`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/utils/_resource.py#L9)
> `ResourceManager(self, time_limit=300, memory_limit=4294967296)`

Resource manager class.

##### Parameters

- `time_limit: int`: maximum amount of seconds a function can run for (default = `5 minutes`).
- `ram_limit: int`: maximum amount of memory bytes a function can use (default = `4 GB`).

##### Notes

- Only one function may be restricted at the same time.
  Upcoming updates will fix this matter.
- Memory restriction is issued upon the process's heap memory size and not
  over total address space in order to get a better estimation of the used memory.
### `get_used_memory`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/utils/_resource.py#L85)
> `get_used_memory(self)`

Returns the amount of memory being used by the current process.
### `run_restricted`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/utils/_resource.py#L100)
> `run_restricted(self, function, *args, **kwargs)`

Executes a given function with restricted amount of
CPU time and RAM memory usage.
### `set_memory_limit`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/utils/_resource.py#L32)
> `set_memory_limit(self, limit)`

Set the memory limit for future restricted functions.

If memory limit is higher or equal than the current OS limit
then it won't be changed.
### `set_time_limit`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/utils/_resource.py#L57)
> `set_time_limit(self, limit)`

