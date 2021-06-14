from typing import Dict


def format_hyperopt_args(args: Dict) -> Dict:
    """
    This method is responsible for formatting hyperopt's args in order to be compatible
    with an ExactSampler.
    """
    new_args = {}
    for key, value in args.items():
        if key == "__CATEGORICAL__":
            inner_key, categorical_value = list(value.items())[0]
            new_args[inner_key] = categorical_value
        elif type(value) == type(dict()):
            new_args[key] = format_hyperopt_args(args[key])
        else:
            new_args[key] = value
    return new_args
