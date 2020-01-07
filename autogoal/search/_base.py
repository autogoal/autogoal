import enlighten

from autogoal.grammar import Grammar


class SearchAlgorithm:
    def __init__(self, grammar: Grammar, fitness_fn, maximize=True):
        self._grammar = grammar
        self._fitness_fn = fitness_fn
        self._maximize = maximize

    def run(self, evaluations, logger=None):
        """Runs the search performing at most `evaluations` of `fitness_fn`.

        Returns:
            Tuple `(best, fn)` of the best found solution and its corresponding fitness.
        """
        if logger is None:
            logger = Looger()

        best_solution = None
        best_fn = None

        logger.begin(evaluations)

        try:
            while evaluations > 0:
                logger.start_generation()
                self._start_generation()

                fns = []

                for solution in self._run_one_generation():
                    logger.sample_solution(solution)
                    fn = self._fitness_fn(solution)
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

        except:
            logger.end(best_solution, best_fn)
            raise

    def _run_one_generation(self):
        raise NotImplementedError()

    def _start_generation(self):
        pass

    def _finish_generation(self, fns):
        pass


class Looger:
    def begin(self):
        pass

    def end(self, best, best_fn):
        pass

    def start_generation(self):
        pass

    def finish_generation(self, fns):
        pass

    def sample_solution(self, solution):
        pass

    def eval_solution(self, solution, fitness):
        pass

    def update_best(self, new_best, new_fn, previous_best, previous_fn):
        pass


class EnlightenLogger(Looger):
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
