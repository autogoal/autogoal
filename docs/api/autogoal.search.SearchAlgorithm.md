# `autogoal.search.SearchAlgorithm`

> [📝](/usr/lib/python3/dist-packages/autogoal/search/_base.py#L15)
> `SearchAlgorithm(self, generator_fn=None, fitness_fn=None, pop_size=1, maximize=True, errors='raise', early_stop=0.5, evaluation_timeout=300, memory_limit=4294967296, search_timeout=3600, target_fn=None, allow_duplicates=True)`

### `run`

> [📝](/usr/lib/python3/dist-packages/autogoal/search/_base.py#L50)
> `run(self, generations=None, logger=None)`

Runs the search performing at most `generations` of `fitness_fn`.

Returns:
    Tuple `(best, fn)` of the best found solution and its corresponding fitness.
