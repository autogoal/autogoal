# `autogoal.search.PESearch`

> [📝](/usr/lib/python3/dist-packages/autogoal/search/_pge.py#L13)
> `PESearch(self, *args, learning_factor=0.05, selection=0.2, epsilon_greed=0.1, random_state=None, name=None, save=True, **kwargs)`

### `load`

> [📝](/usr/lib/python3/dist-packages/autogoal/search/_pge.py#L62)
> `load(self, name_pickle_file)`

Rewrites the probabilistic distribution of metaheuristic with the value of the name model.
        
### `run`

> [📝](/usr/lib/python3/dist-packages/autogoal/search/_base.py#L50)
> `run(self, generations=None, logger=None)`

Runs the search performing at most `generations` of `fitness_fn`.

Returns:
    Tuple `(best, fn)` of the best found solution and its corresponding fitness.
