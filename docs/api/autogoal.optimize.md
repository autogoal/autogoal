# `autogoal.optimize`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/utils/_helpers.py#L46)
> `optimize(fn, search_strategy=<class 'autogoal.search._pge.PESearch'>, generations=100, pop_size=10, allow_duplicates=False, logger=None, **kwargs)`

A general-purpose optimization function.

Simply define any function `fn` with suitable parameter annotations
and apply `optimize`.

##### Parameters

* `search_strategy`: customize the search strategy. By default a `PESearch` will be performed.
* `generations`: max number of generations to run.
* `logger`: instance of `Logger` (or list) to pass to the search strategy.
* `**kwargs`: additional keyword arguments passed to the search strategy constructor.
