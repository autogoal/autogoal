# coding: utf8

from autogoal.grammar import Grammar


class SearchAlgorithm:
    def __init__(self, grammar: Grammar, fitness_fn):
        self._grammar = grammar
        self._fitness_fn = fitness_fn

    def run(self, evaluations):
        """Runs the search performing at most `evaluations` of `fitness_fn`.

        Returns:
            Tuple `(best, fn)` of the best found solution and its corresponding fitness.
        """
        best_solution = None
        best_fn = None

        while evaluations > 0:
            self._start_generation()

            for solution in self._run_one_generation(evaluations):
                evaluations -= 1

                if evaluations <= 0:
                    break

                fn = self._fitness_fn(solution)

                if best_fn is None or fn > best_fn:
                    best_solution = solution
                    best_fn = fn

            self._finish_generation()

        return best_solution, best_fn

    def _run_one_generation(self, max_evaluations):
        raise NotImplementedError()

    def _start_generation(self):
        pass

    def _finish_generation(self):
        pass
