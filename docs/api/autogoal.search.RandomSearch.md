# `autogoal.search.RandomSearch`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/search/_random.py#L7)
> `RandomSearch(self, *args, random_state=None, **kwargs)`

### `run`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/search/_base.py#L48)
> `run(self, evaluations=None, logger=None)`

Runs the search performing at most `evaluations` of `fitness_fn`.

Returns:
    Tuple `(best, fn)` of the best found solution and its corresponding fitness.
