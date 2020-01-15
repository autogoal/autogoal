from ._resource import ResourceManager

import inspect


def nice_repr(cls):
    init_signature = inspect.signature(cls.__init__)
    exclude_param_names = set(['self'])

    def repr_method(self):
        parameter_names = [name for name in init_signature.parameters if name not in exclude_param_names]
        parameter_values = [getattr(self, param, None) for param in parameter_names]
        args = ", ".join(f"{name}={value}" for name, value in zip(parameter_names, parameter_values) if value)

        return f"{cls.__name__}({args})"

    cls.__repr__ = repr_method
    return cls
