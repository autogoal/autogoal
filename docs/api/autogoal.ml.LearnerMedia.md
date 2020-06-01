# `autogoal.ml.LearnerMedia`

> [📝](https://github.com/autogal/autogoal/blob/master/autogoal/ml/_metalearning.py#L163)
> `LearnerMedia(self, problem, solutions, beta=1)`

### `calculate_weight_examples`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/ml/_metalearning.py#L215)
> `calculate_weight_examples(self, solutions)`

Calcule a weight of each example considering the fitness and the similariti with the
actual problem 
### `compute_all_features`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/ml/_metalearning.py#L184)
> `compute_all_features(self)`

### `compute_feature`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/ml/_metalearning.py#L191)
> `compute_feature(self, feature)`

Select for training all solutions where is used the especific feature.

Predict the media of the parameter value.
### `initialize`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/ml/_metalearning.py#L169)
> `initialize(self)`

### `normalize_fitness`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/ml/_metalearning.py#L238)
> `normalize_fitness(self, info)`

Normalize the fitness with respect to the best solution in the problem where that solution is evaluated
        
### `similarity_cosine`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/ml/_metalearning.py#L243)
> `similarity_cosine(self, other_problem)`

Caculate the cosine similarity for a particular solution problem(other problem) 
and the problem analizing
### `similarity_learning`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/ml/_metalearning.py#L252)
> `similarity_learning(self, other_problem)`

Implementar una espicie de encoding para los feature de los problemas
        
