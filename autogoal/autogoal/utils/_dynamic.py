import importlib


def dynamic_import(name, class_name):
    module = importlib.import_module(name)
    return getattr(module, class_name)


def dynamic_call(o: object, attr_name: str, *args, **kwargs):
    attr = getattr(o, attr_name)
    return attr(*args, **kwargs)

def serialize_fitness_fn(fitness_fn):
    import os
    import tempfile
    import inspect

    # Get the source code of the function
    source_code = inspect.getsource(fitness_fn)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp:
        # Write the function to the file
        temp.write(source_code)

        # Make the file hidden
        hidden_file_name = temp.name
        os.rename(temp.name, hidden_file_name)

    print("Hidden finess_fn file created with name:", hidden_file_name)
    return hidden_file_name