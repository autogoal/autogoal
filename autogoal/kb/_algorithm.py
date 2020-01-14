import inspect
from collections import namedtuple


def build_pipelines(input, output, registry):
    all_types = (
        set(_get_annotations(clss).input for clss in registry) | 
        set(_get_annotations(clss).output for clss in registry)
    )
    
    input_types = [t for t in all_types if t.conforms(input)]
    output_types = [t for t in all_types if output.conforms(t)]

    print(all_types)
    print(input_types)
    print(output_types)


Annotations = namedtuple("Annotations", ["input", "output"])


def _get_annotations(clss):
    run_method = clss.run
    input_type = inspect.signature(run_method).parameters['input'].annotation
    output_type = inspect.signature(run_method).return_annotation

    return Annotations(input=input_type, output=output_type)


def _has_input(clss, input):
    return input.conforms(_get_annotations(clss).input)


def _has_output(clss, output):
    return _get_annotations(clss).output.conforms(output)
