# `autogoal.search.PESearch`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/search/_pge.py#L13)
> `PESearch(self, *args, learning_factor=0.05, selection=0.2, epsilon_greed=0.1, random_state=None, name=None, save=True, **kwargs)`

### `load`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/search/_pge.py#L62)
> `load(self, name_pickle_file)`

Rewrites the probabilistic distribution of metaheuristic with the value of the name model.
        
### `run`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/search/_base.py#L48)
> `run(self, evaluations=None, logger=None)`

Runs the search performing at most `evaluations` of `fitness_fn`.

Returns:
    Tuple `(best, fn)` of the best found solution and its corresponding fitness.
