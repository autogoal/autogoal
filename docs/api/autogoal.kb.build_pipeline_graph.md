# `autogoal.kb.build_pipeline_graph`

> [📝](https://github.com/autogoal/autogoal/blob/main/autogoal/kb/_algorithm.py#L74)
> `build_pipeline_graph(input, output, registry, max_list_depth=3, max_pipeline_width=3)`

Creates a `PipelineBuilder` instance that generates all pipelines
from `input` to `output` types.

##### Parameters

- `input`: type descriptor for the desired input.
- `output`: type descriptor for the desired output.
- `registry`: list of available classes to build the pipelines.
