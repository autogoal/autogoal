import importlib


def dynamic_import(name, class_name):
    module = importlib.import_module(name)
    return getattr(module, class_name)


def dynamic_call(o: object, attr_name: str, *args, **kwargs):
    attr = getattr(o, attr_name)
    return attr(*args, **kwargs)
