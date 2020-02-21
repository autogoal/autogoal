# `autogoal.search.SearchAlgorithm`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/search/_base.py#L15)
> `SearchAlgorithm(self, generator_fn=None, fitness_fn=None, pop_size=1, maximize=True, errors='raise', early_stop=None, evaluation_timeout=300, memory_limit=4294967296, search_timeout=3600, target_fn=None)`

### `run`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/search/_base.py#L48)
> `run(self, evaluations=None, logger=None)`

Runs the search performing at most `evaluations` of `fitness_fn`.

Returns:
    Tuple `(best, fn)` of the best found solution and its corresponding fitness.
