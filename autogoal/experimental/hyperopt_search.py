from typing import Dict, cast, List
import math
import time
from hyperopt import fmin, tpe, rand, hp, space_eval, STATUS_OK, STATUS_FAIL
from hyperopt.early_stop import no_progress_loss

from autogoal.experimental.exact_sampler import ExactSampler
from autogoal.grammar import Symbol, generate_cfg
from autogoal.grammar._cfg import (
    Callable,
    Distribution,
    OneOf,
    SubsetOf,
    ContextFreeGrammar,
)
from autogoal.kb._algorithm import PipelineSpace
from autogoal.utils import RestrictedWorkerByJoin, Min, Gb, Sec
from autogoal.search._base import MultiLogger

ALGORITHM_CHOICE_HANDLE = "__CHOICE__"


def cfg_to_hp_space(cfg: ContextFreeGrammar):
    """
    Transforms a `ContextFreeGrammar` in a hyperopt search space
    """
    return _cfg_to_hp_space(cfg, None, None)


def _cfg_to_hp_space(cfg, symbol=None, choice_ref=None):
    start: Symbol = cfg._start if symbol is None else symbol
    node_name = start.name
    grammar_node = cfg[start]
    if type(grammar_node) == Distribution:
        ditribution_name = grammar_node.name
        if ditribution_name == "discrete":
            min_value = grammar_node.parameters["min"] - 0.5
            max_value = grammar_node.parameters["max"] + 0.5
            return hp.quniform(node_name, min_value, max_value, 1)
        elif ditribution_name == "continuous":
            min_value = grammar_node.parameters["min"]
            max_value = grammar_node.parameters["max"]
            return hp.uniform(node_name, min_value, max_value)
        elif ditribution_name == "categorical":
            return hp.choice(
                node_name,
                [{node_name: option} for option in grammar_node.parameters["options"]],
            )
        elif ditribution_name == "boolean":
            return hp.choice(
                node_name, [{node_name: option} for option in [False, True]],
            )

    elif type(grammar_node) == OneOf:
        hp_options = []
        for option in grammar_node.options:
            hp_options.append(_cfg_to_hp_space(cfg, option, node_name))
        return hp.choice(node_name, hp_options)
    elif type(grammar_node) == Callable:
        parameters = {}
        if choice_ref is not None:
            parameters[choice_ref] = start
        for key, value in grammar_node.parameters.items():
            if type(value) == Symbol:
                parameters[value.name] = _cfg_to_hp_space(cfg, value)
            else:
                parameters[key] = _cfg_to_hp_space(cfg, value)
        return parameters
    elif type(grammar_node) == SubsetOf:
        raise NotImplementedError('Hyperopt\'s "subset of" is not yet implemented')


def format_hyperopt_args(args: Dict) -> Dict:
    """
    Transforms arguments in hyperopt's format to a model compatible with ExactSampler
    """
    new_args = {}
    for key, value in args.items():
        if type(value) == type(dict()):
            new_args = {**new_args, **format_hyperopt_args(value)}
        else:
            if key.startswith(ALGORITHM_CHOICE_HANDLE):
                # anything would do, we're just interested in the key here.
                new_args[value] = True
            else:
                new_args[key] = value
    new_args["End"] = True
    return new_args


def pipeline_space_to_hp_space(pipeline_space, registry):
    """
    Transforms a `PipelineSpace` into a hyperopt search space
    """
    return _pipeline_space_to_hp_space(pipeline_space, registry, None, [])


def _pipeline_space_to_hp_space(pipeline_space, registry, start=None, path: List = []):

    if start == pipeline_space.End:
        return None
    graph = pipeline_space.graph
    start_node = start if start is not None else pipeline_space.Start
    node_choices = []
    node_path = path.copy() + [start_node]
    for sub_node in graph[start_node].keys():
        node_choices.append(
            _pipeline_space_to_hp_space(pipeline_space, registry, sub_node, node_path)
        )
    node_choices = list(filter(lambda choice: choice is not None, node_choices))
    if start_node != pipeline_space.Start:
        node_space = cfg_to_hp_space(
            generate_cfg(start_node.algorithm, registry=registry)
        )
        if type(node_space) != type(dict()):
            raise RuntimeError("Unexpected output type for final CFG")
        else:
            node_space = cast(Dict, node_space)
        choice_path = f"{ALGORITHM_CHOICE_HANDLE}{get_path_string(path)}"
        node_space[choice_path] = start_node.algorithm.__name__
    else:
        node_space = {}
    if len(node_choices) > 0:
        path_str = f"{ALGORITHM_CHOICE_HANDLE}{get_path_string(node_path)}"
        node_space[path_str] = hp.choice(
            path_str, [{path_str: option} for option in node_choices]
        )
    return node_space


def get_path_string(path):
    path_strings = [
        repr(node_step.algorithm.__name__)
        if hasattr(node_step, "algorithm")
        else repr(node_step)
        for node_step in path
    ]
    return "->".join(path_strings)


class HyperoptStopException(RuntimeError):
    """ Class for signaling a forced Hyperopt stop 
    """

    ...


class HyperoptSearch:
    def __init__(
        self,
        generator_fn=None,
        fitness_fn=None,
        search_registry=None,
        algorithm=tpe.suggest,
        maximize=True,
        errors="raise",
        early_stop=0.5,
        evaluation_timeout: int = 10 * Sec,
        memory_limit: int = 4 * Gb,
        search_timeout: int = 5 * Min,
        target_fn=None,
        allow_duplicates=True,
        random_state=None,
    ):

        self._generator_fn = generator_fn
        self._fitness_fn = fitness_fn
        self._registry = search_registry
        self._maximize = maximize
        self._errors = errors
        self._evaluation_timeout = evaluation_timeout
        self._memory_limit = memory_limit
        self._early_stop = early_stop
        self._search_timeout = search_timeout
        self._target_fn = target_fn
        self._allow_duplicates = allow_duplicates
        self._logger = None
        self._algo = algorithm

        if self._evaluation_timeout > 0 or self._memory_limit > 0:
            self._fitness_fn = RestrictedWorkerByJoin(
                self._fitness_fn, self._evaluation_timeout, self._memory_limit
            )

    def _hyperopt_fitness(self, args):

        try:
            self._logger.start_generation(self.eval_number, self.best_fn)
            self.eval_number += 1
            fns = []

            improvement = False

            try:
                formatted_args = format_hyperopt_args(args)
                solution = self._generator_fn(ExactSampler(formatted_args))
            except Exception as e:
                self._logger.error("Error while generating solution: %s" % e, args)
                return {"status": STATUS_FAIL, "loss": math.inf}

            if not self._allow_duplicates and repr(solution) in self.seen:
                return {"status": STATUS_OK, "loss": self.seen[repr(solution)]}

            try:
                self._logger.sample_solution(solution)
                fn = self._fitness_fn(solution)
            except Exception as e:
                fn = -math.inf if self._maximize else math.inf
                self._logger.error(e, solution)

                if self._errors == "raise":
                    self._logger.end(self.best_solution, self.best_fn)
                    raise e from None

            if not self._allow_duplicates:
                self.seen[repr(solution)] = fn

            self._logger.eval_solution(solution, fn)
            fns.append(fn)

            if (
                self.best_fn is None
                or (fn > self.best_fn and self._maximize)
                or (fn < self.best_fn and not self._maximize)
            ):
                self._logger.update_best(solution, fn, self.best_solution, self.best_fn)
                self.best_solution = solution
                self.best_fn = fn
                self.improvement = True

                if self._target_fn and self.best_fn >= self._target_fn:
                    raise HyperoptStopException("Target value achieved")

            if not improvement:
                self.no_improvement += 1
            else:
                self.no_improvement = 0

            self._logger.finish_generation(fns)
            loss = -fn if self._maximize else fn
            return {"status": STATUS_OK, "loss": loss}
        except KeyboardInterrupt:
            pass

    def run(self, generations=None, logger=None):
        """Runs the search performing at most `generations` of `fitness_fn`.

        Returns:
            Tuple `(best, fn)` of the best found solution and its corresponding fitness.
        """
        if logger is not None:
            if isinstance(logger, list):
                self._logger = MultiLogger(*logger)
            else:
                self._logger = logger

        if generations is None:
            generations = math.inf

        if isinstance(self._early_stop, float):
            early_stop = int(self._early_stop * generations)
        else:
            early_stop = self._early_stop

        early_stop_fn = no_progress_loss(early_stop) if early_stop is not None else None

        self.best_solution = None
        self.best_fn = None
        self.no_improvement = 0
        self.start_time = time.time()
        self.seen = {}
        self.eval_number = 0

        hp_space = self._generator_fn
        if type(self._generator_fn) == PipelineSpace:
            hp_space = pipeline_space_to_hp_space(self._generator_fn, [])
        if type(self._generator_fn) == ContextFreeGrammar:
            hp_space = cfg_to_hp_space(self._generator_fn)

        self._logger.begin(generations, 1)
        solution = None
        try:
            best_args = fmin(
                self._hyperopt_fitness,
                hp_space,
                self._algo,
                generations,
                self._search_timeout,
                early_stop_fn=early_stop_fn,
                show_progressbar=False,
            )
            best_args = format_hyperopt_args(space_eval(hp_space, best_args))
            solution = self._generator_fn.sample(sampler=ExactSampler(best_args))
        except KeyboardInterrupt:
            pass

        self._logger.end(solution, self.best_fn)
        return solution, self.best_fn
