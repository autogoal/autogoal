# `autogoal.kb.build_pipelines`

> [📝](https://github.com/autogoal/autogoal/blob/master/autogoal/kb/_algorithm.py#L237)
> `build_pipelines(input, output, registry)`

Creates a `PipelineBuilder` instance that generates all pipelines
from `input` to `output` types.

##### Parameters

- `input`: type descriptor for the desired input.
- `output`: type descriptor for the desired output.
- `registry`: list of available classes to build the pipelines.
