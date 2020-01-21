import enlighten
import warnings
import time
import datetime

from autogoal.utils import ResourceManager


class SearchAlgorithm:
    def __init__(self, generator_fn, fitness_fn=None, pop_size=1, maximize=True, errors='raise', evaluation_timeout:int=300, memory_limit:int=4 * 1024 ** 3):
        self._generator_fn = generator_fn
        self._fitness_fn = fitness_fn or self._identity
        self._pop_size = pop_size
        self._maximize = maximize
        self._errors = errors
        self._evaluation_timeout = evaluation_timeout
        self._memory_limit = memory_limit

    def _identity(self, x):
        return x

    def run(self, evaluations, logger=None):
        """Runs the search performing at most `evaluations` of `fitness_fn`.

        Returns:
            Tuple `(best, fn)` of the best found solution and its corresponding fitness.
        """
        if logger is None:
            logger = Logger()

        best_solution = None
        best_fn = None

        logger.begin(evaluations)
        resource_manager = ResourceManager(time_limit = self._evaluation_timeout, memory_limit = self._memory_limit)

        try:
            while evaluations > 0:
                logger.start_generation(evaluations, best_fn)
                self._start_generation()

                fns = []

                for _ in range(self._pop_size):
                    solution = None

                    try:
                        solution = self._generator_fn(self._build_sampler())
                        logger.sample_solution(solution)
                        fn = resource_manager.run_restricted(self._fitness_fn, solution)
                    except Exception as e:
                        fn = 0
                        logger.error(e, solution)

                        if self._errors == 'raise':
                            raise

                    logger.eval_solution(solution, fn)
                    fns.append(fn)

                    if (
                        best_fn is None
                        or (fn > best_fn and self._maximize)
                        or (fn < best_fn and not self._maximize)
                    ):
                        logger.update_best(solution, fn, best_solution, best_fn)
                        best_solution = solution
                        best_fn = fn

                    evaluations -= 1

                    if evaluations <= 0:
                        break

                logger.finish_generation(fns)
                self._finish_generation(fns)

            return best_solution, best_fn

        except KeyboardInterrupt:
            logger.end(best_solution, best_fn)

    def _build_sampler(self):
        raise NotImplementedError()

    def _start_generation(self):
        pass

    def _finish_generation(self, fns):
        pass


class Logger:
    def begin(self, evaluations):
        pass

    def end(self, best, best_fn):
        pass

    def start_generation(self, evaluations, best_fn):
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
    def begin(self, evaluations):
        print("Starting search: evaluations=%i" % evaluations)
        self.start_time = time.time()
        self.start_evaluations = evaluations

    def start_generation(self, evaluations, best_fn):
        current_time = time.time()
        elapsed = int(current_time - self.start_time)
        avg_time = elapsed / (self.start_evaluations - evaluations + 1)
        remaining = int(avg_time * evaluations)
        elapsed = datetime.timedelta(seconds=elapsed)
        remaining = datetime.timedelta(seconds=remaining)
        print("New generation started: best_fn=%.3f, evaluations=%i, elapsed=%s, remaining=%s" % (best_fn or 0, evaluations, elapsed, remaining))

    def error(self, e:Exception, solution):
        print("(!) Error evaluating pipeline: %r" % e)

    def end(self, best, best_fn):
        print("Search completed: best_fn=%.3f, best=\n%r" % (best_fn, best))

    def sample_solution(self, solution):
        print("Evaluating pipeline:\n%r" % solution)

    def eval_solution(self, solution, fitness):
        print("Fitness=%.3f" % fitness)

    def update_best(self, new_best, new_fn, previous_best, previous_fn):
        print("Best solution: improved=%.3f, previous=%.3f" % (new_fn, previous_fn or 0))


class EnlightenLogger(Logger):
    def __init__(self, *, log_solutions=False):
        self.log_solutions = log_solutions

    def begin(self, evaluations):
        self.manager = enlighten.get_manager()
        self.total_counter = self.manager.counter(total=evaluations, unit="runs", leave=False)

    def sample_solution(self, solution):
        if self.log_solutions:
            print(solution)

        self.total_counter.update()

    def eval_solution(self, solution, fn):
        if self.log_solutions:
            print("Fitness: %.4f" % fn)

    def end(self, *args):
        self.total_counter.close()
        self.manager.stop()
