import inspect
import textwrap
import functools

from typing import Callable

from autogoal.search import PESearch
from autogoal.grammar import generate_cfg


class _ParamsDict(dict):
    pass


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
            return _ParamsDict(
                {args_line}
            )
        """
    )

    globals_dict = dict(fn.__globals__)
    globals_dict['_ParamsDict'] = _ParamsDict
    locals_dict = {}
    exec(func_code, globals_dict, locals_dict)
    return locals_dict[func_name]


def optimize(fn, search_strategy=PESearch, generations=100, pop_size=10, allow_duplicates=False, logger=None, **kwargs):
    """
    A general-purpose optimization function.

    Simply define any function `fn` with suitable parameter annotations
    and apply `optimize`.

    ##### Parameters

    * `search_strategy`: customize the search strategy. By default a `PESearch` will be performed.
    * `generations`: max number of generations to run.
    * `logger`: instance of `Logger` (or list) to pass to the search strategy.
    * `**kwargs`: additional keyword arguments passed to the search strategy constructor.
    """
    params_func = _make_params_func(fn)

    @functools.wraps(fn)
    def eval_func(kwargs):
        return fn(**kwargs)

    grammar = generate_cfg(params_func)

    search = search_strategy(grammar, eval_func, pop_size=pop_size, allow_duplicates=allow_duplicates, **kwargs)
    best, best_fn = search.run(generations, logger=logger)

    return best, best_fn
