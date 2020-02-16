# `autogoal.kb`

## Functions

### `algorithm`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/kb/_data.py#L11)
> `algorithm(input_type, output_type)`


!!! warning
    This class has no docstrings.

### `build_composite_list`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/kb/_data.py#L137)
> `build_composite_list(input_type, output_type, depth=1)`


!!! warning
    This class has no docstrings.

### `build_composite_tuple`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/kb/_data.py#L179)
> `build_composite_tuple(index, input_type, output_type)`


Dynamically generate a class `CompositeAlgorithm` that wraps
another algorithm to receive a Tuple but pass only one of the
parameters to the internal algorithm.

### `build_pipelines`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/kb/_algorithm.py#L11)
> `build_pipelines(input, output, registry)`


Creates a `PipelineBuilder` instance that generates all pipelines
from `input` to `output` types.

##### Parameters

- `input`: type descriptor for the desired input.
- `output`: type descriptor for the desired output.
- `registry`: list of available classes to build the pipelines.

### `conforms`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/kb/_data.py#L56)
> `conforms(type1, type2)`


!!! warning
    This class has no docstrings.

### `infer_type`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/kb/_data.py#L239)
> `infer_type(obj)`

Attempts to automatically infer the most precise semantic type for `obj`.

    ##### Parameters

    * `obj`: Object to detect its semantic type.

    ##### Raises

    * `TypeError`: if no valid semantic type was found that matched `obj`.

    ##### Examples

    * Natural language

    ```python
    >>> infer_type("hello")
    Word()
    >>> infer_type("hello world")
    Sentence()
    >>> infer_type("Hello Word. It is raining.")
    Document()

    ```

    * Vectors

    ```
    >>> import numpy as np
    >>> infer_type(np.asarray(["A", "B", "C", "D"]))
    CategoricalVector()
    >>> infer_type(np.asarray([0.0, 1.1, 2.1, 0.2]))
    ContinuousVector()
    >>> infer_type(np.asarray([0, 1, 1, 0]))
    DiscreteVector()

    ```

    * Matrices

    ```
    >>> import numpy as np
    >>> infer_type(np.random.randn(10,10))
    MatrixContinuousDense()

    >>> import scipy.sparse as sp
    >>> infer_type(sp.coo_matrix((10,10)))
    MatrixContinuousSparse()

    ```

