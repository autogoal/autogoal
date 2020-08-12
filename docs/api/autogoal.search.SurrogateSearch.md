# `autogoal.search.SurrogateSearch`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/search/_learning.py#L11)
> `SurrogateSearch(self, base_search, estimator, generation_size=10, initial_pop_size=10, *args, **kwargs)`

### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/search/_base.py#L50)
> `run(self, generations=None, logger=None)`

Runs the search performing at most `generations` of `fitness_fn`.

Returns:
    Tuple `(best, fn)` of the best found solution and its corresponding fitness.
