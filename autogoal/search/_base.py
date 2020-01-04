from autogoal.grammar import Grammar


class SearchAlgorithm:
    def __init__(self, grammar: Grammar, fitness_fn, maximize=True):
        self._grammar = grammar
        self._fitness_fn = fitness_fn
        self._maximize = maximize

    def run(self, evaluations):
        """Runs the search performing at most `evaluations` of `fitness_fn`.

        Returns:
            Tuple `(best, fn)` of the best found solution and its corresponding fitness.
        """
        best_solution = None
        best_fn = None

        while evaluations > 0:
            self._start_generation()

            fns = []

            for solution in self._run_one_generation():
                fn = self._fitness_fn(solution)
                fns.append(fn)

                if (
                    best_fn is None
                    or (fn > best_fn and self._maximize)
                    or (fn < best_fn and not self._maximize)
                ):
                    best_solution = solution
                    best_fn = fn

                evaluations -= 1

                if evaluations <= 0:
                    break

            self._finish_generation(fns)

        return best_solution, best_fn

    def _run_one_generation(self):
        raise NotImplementedError()

    def _start_generation(self):
        pass

    def _finish_generation(self, fns):
        pass
