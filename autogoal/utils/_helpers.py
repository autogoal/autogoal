import inspect
import textwrap
import functools

from typing import Callable

from autogoal.search import PESearch
from autogoal.grammar import generate_cfg


def _make_params_func(fn: Callable):
    signature = inspect.signature(fn)

    func_name = f"{fn.__name__}_params"
    args_names = signature.parameters.keys()

    def annotation_repr(ann):
        if inspect.isclass(ann) or inspect.isfunction(ann):
            return ann.__name__

        return repr(ann)

    args_line = ",\n                ".join(f"{k}={k}" for k in args_names)
    params_line = ", ".join(f"{arg.name}:{annotation_repr(arg.annotation)}" for arg in signature.parameters.values())

    func_code = textwrap.dedent(
        f"""
        def {func_name}({params_line}):
            return dict(
                {args_line}
            )
        """
    )

    locals_dict = {}
    exec(func_code, fn.__globals__, locals_dict)
    return locals_dict[func_name]


def optimize(fn, search_strategy=None, iterations=None, logger=None, **kwargs):
    if search_strategy is None:
        search_strategy = PESearch

    params_func = _make_params_func(fn)

    @functools.wraps(fn)
    def eval_func(kwargs):
        return fn(**kwargs)

    grammar = generate_cfg(params_func)

    search = search_strategy(grammar, eval_func, **kwargs)
    best, best_fn = search.run(iterations, logger=logger)

    return best, best_fn
