import logging
from typing import List, Tuple
from autogoal.search.moo_utils import dominates, non_dominated_sort
import enlighten
import time
import datetime
import statistics
import math
import termcolor
import json
import functools

import autogoal.logging

from autogoal.utils import RestrictedWorkerByJoin, Min, Gb, Sec
from autogoal.sampling import ReplaySampler
from rich.progress import Progress
from rich.panel import Panel


class SearchAlgorithm:
    def __init__(
        self,
        generator_fn=None,
        fitness_fn=None,
        pop_size=20,
        maximize=(True,),
        errors="raise",
        early_stop=0.5,
        evaluation_timeout: int = 10 * Sec,
        memory_limit: int = 4 * Gb,
        search_timeout: int = 5 * Min,
        target_fn=None,
        allow_duplicates=True,
        ranking_fn=None,
    ):
        if generator_fn is None and fitness_fn is None:
            raise ValueError("You must provide either `generator_fn` or `fitness_fn`")

        self._generator_fn = generator_fn
        self._fitness_fn = fitness_fn or (lambda x: x)
        self._pop_size = pop_size
        self._maximize = (
            maximize if isinstance(maximize, (tuple, list)) else (maximize,)
        )
        self._errors = errors
        self._evaluation_timeout = evaluation_timeout
        self._memory_limit = memory_limit
        self._early_stop = early_stop
        self._search_timeout = search_timeout
        self._target_fn = target_fn
        self._allow_duplicates = allow_duplicates

        self._worst_fns = tuple(
            map(lambda to_max: -math.inf if to_max else math.inf, self._maximize)
        )

        if self._evaluation_timeout > 0 or self._memory_limit > 0:
            self._fitness_fn = RestrictedWorkerByJoin(
                self._fitness_fn, self._evaluation_timeout, self._memory_limit
            )

        self._ranking_fn = ranking_fn or (
            lambda _, fns: tuple(
                map(
                    functools.cmp_to_key(
                        lambda x, y: 1
                        if dominates(x, y, self._maximize)
                        else (-1 if dominates(y, x, self._maximize) else 0)
                    ),
                    fns,
                )
            )
        )

    def run(self, generations=None, logger=None):
        """Runs the search performing at most `generations` of `fitness_fn`.

        Returns:
            Tuple `([best1, ..., bestn], [fn1, ..., fnn])` of the best found solutions and their corresponding fitnesses.
        """
        if logger is None:
            logger = Logger()

        if generations is None:
            generations = math.inf

        if isinstance(logger, list):
            logger = MultiLogger(*logger)

        if isinstance(self._early_stop, float):
            early_stop = int(self._early_stop * generations)
        else:
            early_stop = self._early_stop

        no_improvement = 0
        start_time = time.time()
        seen = set()

        # Best solutions for each generation
        best_solutions = []
        best_fns = []

        logger.begin(generations, self._pop_size)

        try:
            while generations > 0:
                stop = False

                # TODO: Update this
                # logger.start_generation(generations, best_fn)
                self._start_generation()

                solutions = []
                fns = []
                for _ in range(self._pop_size):
                    solution = None

                    try:
                        solution = self._generate()
                    except Exception as e:
                        logger.error(
                            "Error while generating solution: %s" % e, solution
                        )
                        continue

                    if not self._allow_duplicates and repr(solution) in seen:
                        continue

                    try:
                        logger.sample_solution(solution)
                        fn = self._fitness_fn(solution)
                    except Exception as e:
                        fn = self._worst_fns
                        logger.error(e, solution)

                        if self._errors == "raise":
                            # TODO: Update this
                            # logger.end(best_solution, best_fn)
                            raise e from None

                    if not self._allow_duplicates:
                        seen.add(repr(solution))

                    logger.eval_solution(solution, fn)
                    solutions.append(solution)
                    fns.append(fn)

                    spent_time = time.time() - start_time
                    if self._search_timeout and spent_time > self._search_timeout:
                        autogoal.logging.logger().info(
                            "(!) Stopping since time spent is %.2f." % (spent_time)
                        )
                        stop = True
                        break

                new_best_solutions, new_best_fns = self._rank_solutions(
                    best_solutions, best_fns, solutions, fns
                )
                if len(best_fns) == 0 or new_best_fns != best_fns:
                    # TODO: Update this
                    # logger.update_best(solution, fn, best_solution, best_fn)
                    best_solutions, best_fns = new_best_solutions, new_best_fns
                    no_improvement = 0
                    if self._target_fn is not None and any(
                        dominates(top_fn, self._target_fn, self._maximize)
                        for top_fn in best_fns
                    ):
                        stop = True
                        break
                else:
                    no_improvement += 1

                generations -= 1
                if generations <= 0:
                    autogoal.logging.logger().info(
                        "(!) Stopping since all generations are done."
                    )
                    # stop = True
                    break

                if early_stop and no_improvement >= early_stop:
                    autogoal.logging.logger().info(
                        "(!) Stopping since no improvement for %i generations."
                        % no_improvement
                    )
                    stop = True
                    break

                logger.finish_generation(fns)
                self._finish_generation(fns)

                if stop:
                    break

        except KeyboardInterrupt:
            pass

        # TODO: Generalize for top solutions
        # logger.end(best_solution, best_fn)
        return best_solutions, best_fns

    def _rank_solutions(self, best_solutions, best_fns, gen_solutions, gen_fns):
        new_best_solutions = list(best_solutions)
        new_best_fns = list(best_fns)
        for solution, fn in zip(gen_solutions, gen_fns):
            if solution is not None:
                new_best_solutions.append(solution)
                new_best_fns.append(fn)

        # No new solutions were found
        if len(new_best_solutions) == len(best_solutions):
            return best_solutions, best_fns

        solutions_ranking = self._ranking_fn(new_best_solutions, new_best_fns)
        _, ranked_solutions, ranked_fns = zip(
            *sorted(
                zip(solutions_ranking, new_best_solutions, new_best_fns),
                key=lambda x: x[0],
                reverse=True,
            )
        )

        optimal_front_indices = non_dominated_sort(ranked_fns, self._maximize)[0]
        optimal_solutions = [ranked_solutions[i] for i in optimal_front_indices]
        optimal_fns = [ranked_fns[i] for i in optimal_front_indices]
        return (optimal_solutions, optimal_fns)

    def _generate(self):
        # BUG: When multiprocessing is used for evaluation and no generation
        #      function is defined, the actual sampling occurs during fitness
        #      evaluation, and since that process has a copy of the solution
        #      we don't get the history in the `ReplaySampler`.

        sampler = ReplaySampler(self._build_sampler())

        if self._generator_fn is not None:
            solution = self._generator_fn(sampler)
        else:
            solution = sampler

        solution.sampler_ = sampler
        return solution

    def _build_sampler(self):
        raise NotImplementedError()

    def _start_generation(self):
        pass

    def _finish_generation(self, fns):
        pass


class Logger:
    def begin(self, generations, pop_size):
        pass

    def end(self, best, best_fn):
        pass

    def start_generation(self, generations, best_fn):
        pass

    def finish_generation(self, fns):
        pass

    def sample_solution(self, solution):
        pass

    def eval_solution(self, solution, fitness):
        pass

    def error(self, e: Exception, solution):
        pass

    def update_best(self, new_best, new_fn, previous_best, previous_fn):
        pass


class ConsoleLogger(Logger):
    def begin(self, generations, pop_size):
        print("Starting search: generations=%i" % generations)
        self.start_time = time.time()
        self.start_generations = generations

    @staticmethod
    def normal(text):
        return termcolor.colored(text, color="gray")

    @staticmethod
    def emph(text):
        return termcolor.colored(text, color="white", attrs=["bold"])

    @staticmethod
    def success(text):
        return termcolor.colored(text, color="green")

    @staticmethod
    def primary(text):
        return termcolor.colored(text, color="blue")

    @staticmethod
    def warn(text):
        return termcolor.colored(text, color="orange")

    @staticmethod
    def err(text):
        return termcolor.colored(text, color="red")

    def start_generation(self, generations, best_fns):
        current_time = time.time()
        elapsed = int(current_time - self.start_time)
        avg_time = elapsed / (self.start_generations - generations + 1)
        remaining = int(avg_time * generations)
        elapsed = datetime.timedelta(seconds=elapsed)
        remaining = datetime.timedelta(seconds=remaining)

        print(
            self.emph("New generation started"),
            *[
                self.success(f"best_fn_{i}={float(best_fn or 0.0):0.3}")
                for i, best_fn in enumerate(best_fns)
            ],
            self.primary(f"generations={generations}"),
            self.primary(f"elapsed={elapsed}"),
            self.primary(f"remaining={remaining}"),
        )

    def error(self, e: Exception, solution):
        print(self.err("(!) Error evaluating pipeline: %s" % e))

    def end(self, best_solutions, best_fns):
        print(
            self.emph("Search completed:"),
            *[
                self.emph(f"best_fn_{i}={best_fn}, best_{i}=\n{best_solution}")
                for i, (best_solution, best_fn) in enumerate(
                    zip(best_solutions, best_fns)
                )
            ],
        )

    def sample_solution(self, solution):
        print(self.emph("Evaluating pipeline:"))
        print(solution)

    def eval_solution(self, solution, fitness):
        print(self.primary("Fitness=%.3f" % fitness))

    def update_best(self, new_best, new_fn, previous_best, previous_fn):
        print(
            self.success(
                "Best solution: improved=%.3f, previous=%.3f"
                % (new_fn, previous_fn or 0)
            )
        )


class ProgressLogger(Logger):
    def begin(self, generations, pop_size):
        self.manager = enlighten.get_manager()
        self.pop_counter = self.manager.counter(
            total=pop_size, unit="evals", leave=True, desc="Current Gen"
        )
        self.total_counter = self.manager.counter(
            total=generations * pop_size, unit="evals", leave=True, desc="Best: 0.000"
        )

    def sample_solution(self, solution):
        self.pop_counter.update()
        self.total_counter.update()

    def start_generation(self, generations, best_fn):
        self.pop_counter.count = 0

    def update_best(self, new_best, new_fn, *args):
        self.total_counter.desc = "Best: %.3f" % new_fn

    def end(self, *args):
        self.pop_counter.close()
        self.total_counter.close()
        self.manager.stop()


class RichLogger(Logger):
    def __init__(self) -> None:
        self.console = autogoal.logging.console()
        self.logger = autogoal.logging.logger()

    def begin(self, generations, pop_size):
        self.progress = Progress(console=self.console)
        self.pop_counter = self.progress.add_task("Generation", total=pop_size)
        self.total_counter = self.progress.add_task(
            "Overall", total=pop_size * generations
        )
        self.progress.start()
        self.console.rule("Search starting", style="blue")

    def sample_solution(self, solution):
        self.progress.advance(self.pop_counter)
        self.progress.advance(self.total_counter)
        self.console.rule("Evaluating pipeline")
        self.console.print(repr(solution))

    def eval_solution(self, solution, fitness):
        self.console.print(Panel(f"📈 Fitness=[blue]{fitness:.3f}"))

    def error(self, e: Exception, solution):
        self.console.print(f"⚠️[red bold]Error:[/] {e}")

    def start_generation(self, generations, best_fns):
        bests = "\n".join(f"Best_{i}: {fn}" for i, fn in enumerate(best_fns))
        self.console.rule(f"New generation - Remaining={generations}\n{bests}")

    def start_generation(self, generations, best_fn):
        self.progress.update(self.pop_counter, completed=0)

    def update_best(self, new_best, new_fn, previous_best, previous_fn):
        self.console.print(
            Panel(
                f"🔥 Best improved from [red bold]{previous_fn or 0:.3f}[/] to [green bold]{new_fn:.3f}[/]"
            )
        )

    def end(self, best, best_fn):
        self.console.rule(f"Search finished")
        self.console.print(repr(best))
        self.console.print(Panel(f"🌟 Best=[green bold]{best_fn or 0:.3f}"))
        self.progress.stop()
        self.console.rule("Search finished", style="red")


class JsonLogger(Logger):
    def __init__(self, log_file_name: str) -> None:
        self.log_file_name = log_file_name
        with open(self.log_file_name, "w") as log_file:
            print("creating log file ", self.log_file_name)
            json.dump([], log_file)

    def begin(self, generations, pop_size):
        pass

    def start_generation(self, generations, best_fn):
        eval_log = {"generations left": generations, "best_fn": best_fn}
        self.update_log(eval_log)

    def update_best(self, new_best, new_fn, previous_best, previous_fn):
        eval_log = {"new-best-fn": new_fn, "previous-best-fn": previous_fn}
        self.update_log(eval_log)

    def eval_solution(self, solution, fitness):
        eval_log = {
            "pipeline": repr(solution)
            .replace("\n", "")
            .replace(" ", "")
            .replace(",", ", "),
            "multiline-pipeline": repr(solution),
            "fitness": fitness,
        }
        self.update_log(eval_log)

    def end(self, best, best_fn):
        eval_log = {
            "Finished run": True,
            "best-pipeline": repr(best)
            .replace("\n", "")
            .replace(" ", "")
            .replace(",", ", "),
            "best-multiline-pipeline": repr(best),
            "best-fitness": best_fn,
        }
        self.update_log(eval_log)

    def update_log(self, json_load):
        new_data = ""
        with open(self.log_file_name, "r") as log_file:
            data = json.load(log_file)
            new_data = data
            new_data.append(json_load)

        with open(self.log_file_name, "w") as log_file:
            json.dump(new_data, log_file)


class MemoryLogger(Logger):
    def __init__(self):
        self.generation_best_fn = [0]
        self.generation_mean_fn = []

    def update_best(self, new_best, new_fn, previous_best, previous_fn):
        self.generation_best_fn[-1] = new_fn

    def finish_generation(self, fns):
        try:
            mean = statistics.mean(fns)
        except:
            mean = 0
        self.generation_mean_fn.append(mean)
        self.generation_best_fn.append(self.generation_best_fn[-1])


class MultiLogger(Logger):
    def __init__(self, *loggers):
        self.loggers = loggers

    def run(self, name, *args, **kwargs):
        for logger in self.loggers:
            getattr(logger, name)(*args, **kwargs)

    def begin(self, *args, **kwargs):
        self.run("begin", *args, **kwargs)

    def end(self, *args, **kwargs):
        self.run("end", *args, **kwargs)

    def start_generation(self, *args, **kwargs):
        self.run("start_generation", *args, **kwargs)

    def finish_generation(self, *args, **kwargs):
        self.run("finish_generation", *args, **kwargs)

    def sample_solution(self, *args, **kwargs):
        self.run("sample_solution", *args, **kwargs)

    def eval_solution(self, *args, **kwargs):
        self.run("eval_solution", *args, **kwargs)

    def error(self, *args, **kwargs):
        self.run("error", *args, **kwargs)

    def update_best(self, *args, **kwargs):
        self.run("update_best", *args, **kwargs)
