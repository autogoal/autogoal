# `autogoal.search.RandomSearch`

> [📝](https://github.com/autogal/autogoal/blob/main/autogoal/search/_random.py#L7)
> `RandomSearch(self, *args, random_state=None, **kwargs)`

### `run`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/search/_base.py#L50)
> `run(self, generations=None, logger=None)`

Runs the search performing at most `generations` of `fitness_fn`.

Returns:
    Tuple `(best, fn)` of the best found solution and its corresponding fitness.
